import math
import operator
import sys
import os
from os.path import dirname, abspath
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage.util.dtype import dtype_range

from kornia.morphology import dilation
from kornia.morphology import erosion
from skimage.filters import threshold_otsu
from skimage import measure

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.cams.core_seeding import STOtsu

from dlib.configure import constants
from dlib.utils.wsol import compute_bboxes_from_scoremaps_ext_contours
from dlib.utils.wsol import check_box_convention

from dlib.cams.decay_temp import DecayTemp

__all__ = ['TCAMSeeder', 'GetRoiSingleCam']


def get_largest_bbox(bboxes: np.ndarray) -> np.ndarray:
    assert bboxes.ndim == 2, bboxes.ndim
    assert bboxes.shape[1] == 4, bboxes.shape[1]

    out_bbox = None
    area = 0.0
    for i in range(bboxes.shape[0]):
        bb = bboxes[i].reshape(1, -1)
        check_box_convention(bb, 'x0y0x1y1')
        widths = bb[0, 2] - bb[0, 0]
        heights = bb[0, 3] - bb[0, 1]
        c_area = widths * heights
        if c_area >= area:
            area = c_area
            out_bbox = bb

    return out_bbox


class TCAMSeeder(nn.Module):
    def __init__(self,
                 seed_tech: str,
                 min_: int,
                 max_: int,
                 max_p: float,
                 min_p: float,
                 fg_erode_k: int,
                 fg_erode_iter: int,
                 ksz: int,
                 support_background: bool,
                 multi_label_flag: bool,
                 seg_ignore_idx: int,
                 cuda_id: int,
                 roi_method: str,
                 p_min_area_roi: float,
                 use_roi: bool
                 ):
        super(TCAMSeeder, self).__init__()

        assert seed_tech in constants.SEED_TECHS, seed_tech
        self.seed_tech = seed_tech

        # todo: remove: roi_method: str, p_min_area_roi: float,

        assert not multi_label_flag

        assert isinstance(cuda_id, int)
        assert cuda_id >= 0, cuda_id
        self._device = torch.device(cuda_id)

        assert isinstance(ksz, int)
        assert ksz > 0
        self.ksz = ksz
        self.kernel = None
        if self.ksz > 1:
            self.kernel = torch.ones((self.ksz, self.ksz), dtype=torch.long,
                                     device=self._device)

        assert isinstance(min_, int)
        assert isinstance(max_, int)
        assert min_ >= 0
        assert max_ >= 0
        assert min_ + max_ > 0

        self.min_ = min_
        self.max_ = max_

        assert isinstance(min_p, float)
        assert 0. <= min_p <= 1.
        self.min_p = min_p

        assert isinstance(max_p, float)
        assert 0. <= max_p <= 1.
        self.max_p = max_p

        # fg
        assert isinstance(fg_erode_k, int)
        assert fg_erode_k >= 1
        self.fg_erode_k = fg_erode_k

        # fg
        assert isinstance(fg_erode_iter, int)
        assert fg_erode_iter >= 0
        self.fg_erode_iter = fg_erode_iter

        self.fg_kernel_erode = None
        if self.fg_erode_iter > 0:
            assert self.fg_erode_k > 1
            self.fg_kernel_erode = torch.ones(
                (self.fg_erode_k, self.fg_erode_k), dtype=torch.long,
                device=self._device)

        self.support_background = support_background
        self.multi_label_flag = multi_label_flag

        self.ignore_idx = seg_ignore_idx

        assert roi_method in constants.ROI_SELECT, roi_method
        self.roi_method: str = roi_method
        assert 0. < p_min_area_roi < 1., p_min_area_roi
        self.p_min_area_roi: float = p_min_area_roi

        self.use_roi: bool = use_roi

    def set_seed_tech(self, seed_tech):
        assert seed_tech in constants.SEED_TECHS, seed_tech
        self.seed_tech = seed_tech

    def mb_erosion_n(self, x):
        assert self.fg_erode_k > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = x

        for i in range(self.fg_erode_iter):
            out = erosion(out, self.fg_kernel_erode)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def mb_dilate(self, x):
        assert self.ksz > 1
        assert x.ndim == 4
        assert x.shape[1] == 1

        out: torch.Tensor = dilation(x, self.kernel)

        # expensive
        # assert 0 <= torch.min(out) <= 1
        # assert 0 <= torch.max(out) <= 1

        assert out.shape == x.shape

        return out

    def identity(self, x):
        return x

    def forward(self,
                x: torch.Tensor,
                roi: torch.Tensor = None) -> torch.Tensor:

        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        if roi is not None:
            assert torch.is_tensor(roi)
            assert roi.ndim == 4  # b, 1, h, w
            assert roi.shape[0] == x.shape[0], f'{roi.shape[0]}, {x.shape[0]}'
            assert roi.shape[1] == 1, roi.shape[1]

            assert roi.shape[2:] == x.shape[2:]

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1, d  # todo multilabel.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        nbr_bg = int(self.min_p * h * w)

        all_fg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)
        all_bg = torch.zeros((b, h, w), dtype=torch.long, device=x.device,
                             requires_grad=False)

        opx = _OneSample(min_p=self.min_p,
                         max_p=self.max_p,
                         min_=self.min_,
                         max_=self.max_,
                         seed_tech=self.seed_tech,
                         roi_method=self.roi_method,
                         p_min_area_roi=self.p_min_area_roi,
                         use_roi=self.use_roi
                         )

        for i in range(b):
            _roi = None
            if roi is not None:
                _roi = roi[i].squeeze()  # h, w. long
            all_fg[i], all_bg[i] = opx(cam=x[i].squeeze(), erode=erode,
                                       roi=_roi)

        # fg
        all_fg = dilate(all_fg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # bg
        all_bg = dilate(all_bg.unsqueeze(1)).squeeze(1)
        # b, h, w

        # sanity
        outer = all_fg + all_bg
        all_fg[outer == 2] = 0
        all_bg[outer == 2] = 0

        # assign
        out[all_fg == 1] = 1
        out[all_bg == 1] = 0

        out = out.detach()

        assert out.dtype == torch.long

        return out

    def use_all_roi(self,
                    x: torch.Tensor,
                    roi: torch.Tensor = None) -> torch.Tensor:

        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

        assert roi is not None
        assert torch.is_tensor(roi)
        assert roi.ndim == 4  # b, 1, h, w
        assert roi.shape[0] == x.shape[0], f'{roi.shape[0]}, {x.shape[0]}'
        assert roi.shape[1] == 1, roi.shape[1]

        assert roi.shape[2:] == x.shape[2:]

        # expensive
        # assert 0 <= torch.min(x) <= 1
        # assert 0 <= torch.max(x) <= 1

        if self.ksz == 1:
            dilate = self.identity
        elif self.ksz > 1:
            dilate = self.mb_dilate
        else:
            raise ValueError

        if self.fg_erode_iter > 0:
            erode = self.mb_erosion_n
        else:
            erode = self.identity

        b, d, h, w = x.shape
        assert d == 1, d  # todo multilabel.

        out = torch.zeros((b, h, w), dtype=torch.long, requires_grad=False,
                          device=self._device) + self.ignore_idx

        _roi = roi.squeeze(1)  # b, h, w
        out[_roi == 1] = 1
        out = out.detach()

        assert out.dtype == torch.long

        return out

    def extra_repr(self):
        return f'min_={self.min_}, max_={self.max_}, min_p={self.min_p},' \
               f'max_p={self.max_p}, ksz={self.ksz}, fg_erode_k: ' \
               f'{self.fg_erode_k}, fg_erode_iter: {self.fg_erode_iter}' \
               f'support_background={self.support_background},' \
               f'multi_label_flag={self.multi_label_flag}, ' \
               f'seg_ignore_idx={self.ignore_idx}, seed_tech={self.seed_tech}'


class GetRoiSingleCam(object):
    def __init__(self, roi_method: str, p_min_area_roi: float):

        assert roi_method in constants.ROI_SELECT, roi_method
        self.roi_method: str = roi_method
        assert 0 < p_min_area_roi < 1., p_min_area_roi
        self.p_min_area_roi = p_min_area_roi  # min area percentage
        # considered roi

    def __call__(self,
                 cam: torch.Tensor,
                 thresh: float = None) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:

        # thresh in [0, 255].
        assert torch.is_tensor(cam)
        assert cam.ndim == 2, cam.ndim  # h,w

        _cam = cam.cpu().detach().numpy()
        h, w = _cam.shape

        if thresh is None:
            _thresh = self.get_thresh(_cam)  # in [0, 255]
        else:
            assert thresh >= 0, thresh  # in [0, 1.]
            _thresh = thresh * 255.


        blobs = (_cam * 255. >= _thresh).astype(int)

        bbox = np.array([0, 0, h - 1, w - 1]).reshape((1, 4))  # not used.

        if self.roi_method == constants.ROI_ALL:
            final_roi = blobs

        elif self.roi_method in [constants.ROI_H_DENSITY,
                                 constants.ROI_LARGEST]:

            blobs_labels = measure.label(blobs, background=0, connectivity=1,
                                         return_num=False)

            labels = np.unique(blobs_labels)
            if labels.size == 1:
                final_roi = blobs.astype(np.float)
            else:

                label_density = dict()
                min_area = (h * w) * self.p_min_area_roi
                label_area = dict()

                for l in labels:
                    if l == 0:
                        continue

                    s_roi = (blobs_labels == l).astype(float)
                    s_cam = _cam * s_roi
                    s_roi_area = s_roi.sum()
                    label_density[l] = s_cam.sum() / s_roi_area
                    label_area[l] = s_roi_area

                if self.roi_method == constants.ROI_H_DENSITY:

                    l_roi = max(label_density, key=label_density.get)

                    if label_area[l_roi] < min_area:
                        l_roi = max(label_area, key=label_area.get)

                elif self.roi_method == constants.ROI_LARGEST:
                    l_roi = max(label_area, key=label_area.get)

                else:
                    raise NotImplementedError(self.roi_method)

                final_roi = (blobs_labels == l_roi).astype(float)

            # bbox

            l_bbox, nbr_bbox = compute_bboxes_from_scoremaps_ext_contours(
                scoremap=final_roi, scoremap_threshold_list=[0.5],
                multi_contour_eval=True, bbox=None)  # one bbox only.

            assert len(l_bbox) == 1
            largest_bbox = get_largest_bbox(bboxes=l_bbox[0])
            if largest_bbox is not None:  # (1, 4). np.ndarary,x0y0x1y1
                assert largest_bbox.shape == (1, 4), largest_bbox.shape
            else:
                largest_bbox = np.array([0, 0, h - 1, w - 1]).reshape((1, 4))

            bbox = largest_bbox

        else:
            raise NotImplementedError(self.roi_method)

        bbox_mask = np.zeros((h, w), dtype=np.float32)
        x0, y0, x1, y1 = bbox.flatten()
        bbox_mask[y0:y1, x0:x1] = 1.
        bbox_mask = torch.from_numpy(bbox_mask).float()  # cpu, float, h, w.
        bbox = torch.from_numpy(bbox).float()  # cpu, float, (1, 4)

        final_roi = torch.tensor(final_roi, dtype=torch.long,
                            requires_grad=False) # cpu. long. h, w

        return final_roi, bbox_mask, bbox

    @staticmethod
    def get_thresh(cam: np.ndarray) -> float:
        cam_ = np.floor(cam * 255.)

        if cam_.min() == cam_.max():
            th = 0.
        else:
            th = threshold_otsu(cam_)

        return th


class _OneSample(nn.Module):
    def __init__(self,
                 min_p: float,
                 max_p: float,
                 min_: int,
                 max_: int,
                 seed_tech: str,
                 roi_method: str,
                 p_min_area_roi: float,
                 use_roi: bool):
        super(_OneSample, self).__init__()

        self.use_roi = use_roi
        self.otsu = STOtsu()
        self.get_roi = GetRoiSingleCam(roi_method=roi_method,
                                       p_min_area_roi=p_min_area_roi)
        self.fg_capture = _SFG(max_p=max_p, max_=max_, seed_tech=seed_tech)
        self.bg_capture = _SBG(min_p=min_p, min_=min_,
                               seed_tech=constants.SEED_UNIFORM)

    def forward(self,
                cam: torch.Tensor,
                erode: Callable[[torch.Tensor], torch.Tensor],
                roi: torch.Tensor = None
                ) -> Tuple[
        torch.Tensor, torch.Tensor]:

        assert cam.ndim == 2, cam.ndim
        if roi is not None:
            assert roi.ndim == 2, roi.ndim

        # cam: h, w
        h, w = cam.shape
        fg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)
        bg = torch.zeros((h, w), dtype=torch.long, device=cam.device,
                         requires_grad=False)

        if cam.min() == cam.max():
            return fg, bg

        # ROI
        _roi = roi
        if self.use_roi:
            if roi is None:
                _roi, _, _ = self.get_roi(cam, thresh=None)  # cpu, long. h,w
                _roi = _roi.to(cam.device)

            _roi = erode(_roi.unsqueeze(0).unsqueeze(0)).squeeze()
        else:
            _roi = None

        fg = self.fg_capture(cam=cam, roi=_roi, fg=fg)
        bg = self.bg_capture(cam=cam, bg=bg)
        return fg, bg


class _SFG(nn.Module):
    def __init__(self, max_p: float, max_: int, seed_tech: str):
        super(_SFG, self).__init__()

        self.max_ = max_
        self.max_p = max_p
        self.seed_tech = seed_tech

    def forward(self,
                cam: torch.Tensor,
                roi: torch.Tensor,
                fg: torch.Tensor = None) -> torch.Tensor:

        h, w = cam.shape

        if roi is not None:
            assert cam.shape == roi.shape
            n = roi.sum()
            _cam = cam * roi
            _cam = _cam + 1e-8
        else:
            n = h * w
            _cam = cam + 1e-8

        n = int(self.max_p * n)
        _cam_flatten = _cam.view(h * w)

        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=True,
                               stable=True)

        if (n > 0) and (self.max_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == constants.SEED_UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == constants.SEED_WEIGHTED:
                probs = _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                # probs = torch.exp(probs * 10)
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.max_, n), replacement=False)

            fg[_idx[0][selected], _idx[1][selected]] = 1

        return fg


class _SBG(nn.Module):
    def __init__(self, min_p: float, min_: int, seed_tech: str):
        super(_SBG, self).__init__()

        self.min_ = min_
        self.min_p = min_p
        self.seed_tech = seed_tech

    def forward(self, cam: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        assert cam.ndim == 2

        # cam: h, w
        h, w = cam.shape

        n = int(self.min_p * h * w)

        _cam = cam + 1e-8
        _cam_flatten = _cam.view(h * w)
        val, idx_ = torch.sort(_cam_flatten, dim=0, descending=False,
                               stable=True)

        if (n > 0) and (self.min_ > 0):

            tmp = _cam_flatten * 0.
            tmp[idx_[:n]] = 1
            tmp = tmp.view(h, w)

            _idx = torch.nonzero(tmp, as_tuple=True)  # (idx, idy)

            if self.seed_tech == constants.SEED_UNIFORM:
                probs = torch.ones(n, dtype=torch.float, device=cam.device)

            elif self.seed_tech == constants.SEED_WEIGHTED:
                probs = 1. - _cam[_idx[0], _idx[1]]  # 1d array. numel: n
                probs = torch.relu(probs) + 1e-8
                assert probs.numel() == n

            else:
                raise NotImplementedError(self.seed_tech)

            selected = probs.multinomial(
                num_samples=min(self.min_, n),
                replacement=False)
            bg[_idx[0][selected], _idx[1][selected]] = 1

        return bg


def test_TCAMSeeder():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time
    from os.path import join

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'y'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return (_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def plot_limgs(_lims, title):
        nrows = 1
        ncols = len(_lims)

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        for i, (im, tag) in enumerate(_lims):
            if tag == 'CAM':
                axes[0, i].imshow(im, cmap='jet')
            else:
                axes[0, i].imshow(im, cmap=get_cm())

            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    cuda = 1
    device = torch.device(f'cuda:{cuda}')
    torch.cuda.set_device(cuda)

    seed = 0
    min_ = 1
    max_ = 1
    min_p = 0.1
    max_p = 0.7
    fg_erode_k = 11
    fg_erode_iter = 0
    ksz = 3

    batchs = 1

    cam = torch.rand((batchs, 1, 224, 224), dtype=torch.float,
                     device=torch.device('cuda'), requires_grad=False)
    cam = cam * 0

    path_cam = join(root_dir,
                    'data/debug/input/train_n02229544_n02229544_2648.JPEG.npy')
    cam = np.load(path_cam).squeeze()
    cam: torch.Tensor = torch.from_numpy(cam).to(device).float()
    cam.requires_grad = False
    cam = cam.view(1, 224, 224).repeat(batchs, 1, 1, 1)

    # for i in range(batchs):
    #     cam[i, 0, 100:150, 100:150] = 1

    limgs_conc = [(cam_2Img(cam), 'CAM')]

    for seed_tech in [constants.SEED_UNIFORM, constants.SEED_WEIGHTED]:
        set_seed(seed)
        module_conc = TCAMSeeder(
            seed_tech=seed_tech,
            min_=min_,
            max_=max_,
            min_p=min_p,
            max_p=max_p,
            fg_erode_k=fg_erode_k,
            fg_erode_iter=fg_erode_iter,
            ksz=ksz,
            support_background=True,
            multi_label_flag=False,
            seg_ignore_idx=-255,
            cuda_id=cuda,
            roi_method=constants.ROI_ALL,
            p_min_area_roi=5/100.,
            use_roi=False
        )
        announce_msg('Testing {}'.format(module_conc))

        # cProfile.runctx('module_conc(cam)', globals(), locals())

        t0 = time.perf_counter()
        out = module_conc(cam)
        out[out == 1] = 255
        out[out == 0] = int(255 / 2)
        out[out == module_conc.ignore_idx] = 0
        tt = 100
        for _ in range(tt):
            _out = module_conc(cam)
            _out[_out == 1] = 255
            _out[_out == 0] = int(255 / 2)
            _out[_out == module_conc.ignore_idx] = 0
            out += _out

        out = (out).long()
        print('time CONC: {}s'.format(time.perf_counter() - t0))

        # out[out > (255/2)] = 255
        # out[(0 <= out) & (out <= (255/2))] = int(255 / 2)
        # out[out == module_conc.ignore_idx] = 0

        if batchs == 1:
            limgs_conc.append((out.squeeze().cpu().numpy().astype(np.uint8),
                               'S_{}'.format(seed_tech)))

    if batchs == 1:
        plot_limgs(limgs_conc, 'Seeding techs')

def test_GetRoiSingleCam():
    import cProfile

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time
    from os.path import join

    _ARBITRARY_COLOR = mcolors.CSS4_COLORS['orange']

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'y'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return (_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def convert_bbox(bbox_xyxy: np.ndarray):
        check_box_convention(bbox_xyxy, 'x0y0x1y1')
        assert bbox_xyxy.shape == (1, 4), bbox_xyxy.shape
        x0, y0, x1, y1 = bbox_xyxy.flatten()
        width = x1 - x0
        height = y1 - y0
        anchor = (x0, y1)
        return anchor, width, height

    def plot_limgs(_lims, title):

        bbo_info = None
        for i, (cnt, tag) in enumerate(_lims):
            if tag == 'bbox':
                bbo_info = convert_bbox(cnt)


        nrows = 1
        ncols = len(_lims)

        if bbo_info is not None:
            ncols = ncols - 1
            tmp = [el for el in _lims if el[1] != 'bbox']
            _lims = tmp

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 10
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))

        for i, (im, tag) in enumerate(_lims):
            if tag == 'CAM':
                axes[0, i].imshow(im, cmap='jet')
            else:
                axes[0, i].imshow(im, cmap=get_cm())

            if bbo_info is not None:
                rect_roi = patches.Rectangle(bbo_info[0], bbo_info[1],
                                             -bbo_info[2],
                                             linewidth=3.,
                                             edgecolor=_ARBITRARY_COLOR,
                                             facecolor='none')
                axes[0, i].add_patch(rect_roi)

            axes[0, i].text(3, 40, tag,
                            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
        plt.suptitle(title)
        plt.show()

    seed = 0
    path_cam = join(root_dir,
                    'data/debug/input/train_n02229544_n02229544_2648.JPEG.npy')
    cam = np.load(path_cam).squeeze()
    cam = torch.from_numpy(cam)

    for roi_method in constants.ROI_SELECT:
        limgs_conc = [(cam_2Img(cam), 'CAM')]
        th = threshold_otsu(cam.numpy())
        limgs_conc.append((cam_2Img((cam >= th).long()), 'ALL-ROIs'))

        set_seed(seed)
        module = GetRoiSingleCam(roi_method=roi_method, p_min_area_roi=5/100.)
        announce_msg('Testing {}'.format(module))

        t0 = time.perf_counter()
        final_roi, bbox_mask, bbox = module(cam)
        print('time: {}s'.format(time.perf_counter() - t0))

        final_roi[final_roi == 1] = 255
        final_roi[final_roi == 0] = int(255 / 2)

        limgs_conc.append((final_roi.squeeze().cpu().numpy().astype(np.uint8),
                           'SELECTED ROI'))

        limgs_conc.append((bbox_mask.squeeze().cpu().numpy().astype(np.uint8),
                           'mask'))

        limgs_conc.append((bbox.cpu().numpy(), 'bbox'))

        plot_limgs(limgs_conc, roi_method)


def test_CAM_distribution():
    import cProfile

    from dlib.utils.shared import find_files_pattern

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    from torch.profiler import profile, record_function, ProfilerActivity
    import time
    from os.path import join

    _ARBITRARY_COLOR = mcolors.CSS4_COLORS['orange']

    _CAM = 'cam'
    _HIST = 'histo'

    def get_cm():
        col_dict = dict()
        for i in range(256):
            col_dict[i] = 'k'

        col_dict[0] = 'k'
        col_dict[int(255 / 2)] = 'y'
        col_dict[255] = 'r'
        colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

        return colormap

    def cam_2Img(_cam):
        return (_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def convert_bbox(bbox_xyxy: np.ndarray):
        check_box_convention(bbox_xyxy, 'x0y0x1y1')
        assert bbox_xyxy.shape == (1, 4), bbox_xyxy.shape
        x0, y0, x1, y1 = bbox_xyxy.flatten()
        width = x1 - x0
        height = y1 - y0
        anchor = (x0, y1)
        return anchor, width, height

    def avg_cams(lcams: list):
        out = None
        for cam in lcams:
            if out is None:
                out = cam
            else:
                out = out + cam

        return out / float(len(lcams))

    def max_cams(lcams: list):
        out = None
        for cam in lcams:
            if out is None:
                out = cam
            else:
                out = torch.max(out, cam)

        return out

    def renormalize(lcams: list, f: float):
        out = []
        for i, cam in enumerate(lcams):
            e = torch.exp(cam * f)
            out.append(e / e.max())

        return out

    def plot_limgs(_lims, title, typefig=_CAM):

        bbo_info = None
        for i, (cnt, tag) in enumerate(_lims):
            if tag == 'bbox':
                bbo_info = convert_bbox(cnt)


        if len(_lims) <= 4:
            nrows = 1
            ncols = len(_lims)
        else:
            nrows = 4
            ncols = math.ceil(len(_lims) / nrows)

        if bbo_info is not None:
            ncols = ncols - 1
            tmp = [el for el in _lims if el[1] != 'bbox']
            _lims = tmp

        him, wim = _lims[0][0].shape
        r = him / float(wim)
        fw = 20
        r_prime = r * (nrows / float(ncols))
        fh = r_prime * fw

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False, figsize=(fw, fh))
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                if k >= len(_lims):
                    axes[i, j].set_visible(False)
                    k += 1
                    continue

                im, tag = _lims[k]

                if tag.startswith('CAM'):
                    if typefig == _CAM:
                        axes[i, j].imshow(im, cmap='jet')
                    elif typefig == _HIST:
                        axes[i, j].hist(im, 50, density=True, facecolor='g',
                                        alpha=0.75)
                else:
                    axes[i, j].imshow(im, cmap=get_cm())

                if bbo_info is not None:
                    rect_roi = patches.Rectangle(bbo_info[0], bbo_info[1],
                                                 -bbo_info[2],
                                                 linewidth=3.,
                                                 edgecolor=_ARBITRARY_COLOR,
                                                 facecolor='none')
                    axes[i, j].add_patch(rect_roi)

                axes[i, j].text(3, 40, tag,
                                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})

                k += 1

        plt.suptitle(title)
        plt.show()

    seed = 0
    fd_cam = join(root_dir, 'data/debug/input/cams')
    files = find_files_pattern(fd_cam, '*.pt')

    lcams = [torch.load(x).squeeze() for x in files]
    limgs_conc = [(cam_2Img(cam), f'CAM-{i}') for i, cam in enumerate(lcams)]
    # lhist = [(cam.numpy(), f'CAM-{i}') for i, cam in enumerate(lcams)]
    limgs_conc.append((cam_2Img(avg_cams(lcams)), 'CAM-avg'))
    limgs_conc.append((cam_2Img(max_cams(lcams)), 'CAM-max'))
    # lhist.append((avg_cams(lcams).numpy(), 'CAM-avg'))
    # lhist.append((max_cams(lcams).numpy(), 'CAM-max'))
    plot_limgs(limgs_conc, 'aggrg. no renormalization', _CAM)
    # plot_limgs(lhist, 'agr. no renormalization', _HIST)


    # with renormalization

    for heat in range(1, 21, 1):
        lcams = [torch.load(x).squeeze() for x in files]
        heat = heat / 10.
        lcams = renormalize(lcams, heat)
        limgs_conc = [(cam_2Img(cam), f'CAM-{i}') for i, cam in enumerate(lcams)]
        limgs_conc.append((cam_2Img(avg_cams(lcams)), 'CAM-avg'))
        limgs_conc.append((cam_2Img(max_cams(lcams)), 'CAM-max'))
        plot_limgs(limgs_conc, f'aggr. with renormalization. Heat: {heat}', _CAM)

if __name__ == '__main__':
    import datetime as dt

    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    # set_seed(0)
    # test_TCAMSeeder()

    # set_seed(0)
    # test_GetRoiSingleCam()

    set_seed(0)
    test_CAM_distribution()
