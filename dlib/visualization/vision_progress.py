import copy
import os
import sys
from os.path import dirname, abspath, join
from typing import Tuple, Union

from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
from skimage.filters import threshold_otsu

import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.cams import build_std_cam_extractor
from dlib.cams import build_fcam_extractor
from dlib.cams import build_tcam_extractor

from dlib.utils.tools import t2n
from dlib.utils.wsol import check_scoremap_validity
from dlib.utils.wsol import check_box_convention
from dlib.utils.shared import reformat_id
from dlib.visualization.vision_wsol import Viz_WSOL

from dlib.utils.wsol import compute_bboxes_from_scoremaps_ext_contours
from dlib.datasets.wsol_loader import get_eval_tranforms

from dlib.configure import constants
import dlib.dllogger as DLLogger


__all__ = ['plot_progress_cams']


def _build_std_cam_extractor(classifier, args):
    return build_std_cam_extractor(classifier=classifier, args=args)


def _build_fcam_extractor(model, args):
    return build_fcam_extractor(model=model, args=args)


def _build_tcam_extractor(model, args):
    return build_tcam_extractor(model=model, args=args)

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



def get_cam_one_sample(args,
                       model,
                       cam_extractor,
                       image: torch.Tensor,
                       target: int,
                       blured_img: torch.Tensor = None
                       ) -> Tuple[torch.Tensor, torch.Tensor,
                                  Union[dict, None]]:
    cam, cbox = None, None
    task = args.task
    fcam_argmax = False

    with autocast(enabled=args.amp_eval):
        output = model(image)

    if task == constants.STD_CL:

        if args.amp_eval:
            output = output.float()

        cl_logits = output
        cam = cam_extractor(class_idx=target,
                            scores=cl_logits,
                            normalized=True)

        # (h`, w`)

    elif task in [constants.F_CL, constants.TCAM]:

        if args.amp_eval:
            tmp = []
            for term in output:
                tmp.append(term.float() if term is not None else None)
            output = tmp

        cl_logits, fcams, im_recon = output
        if task == constants.F_CL:
            cam = cam_extractor(argmax=fcam_argmax)
            # (h`, w`)
        elif task == constants.TCAM:
            cam = cam_extractor(argmax=fcam_argmax)
            # (h`, w`)

    elif args.task == constants.C_BOX:
        raise NotImplementedError
    else:
        raise NotImplementedError

    if cam is not None:
        if args.amp_eval:
            cam = cam.float()

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        # cl_logits: 1, nc.

    return cam, cl_logits, cbox


def get_str_trg_prd_cl(pred_cl: int, trg_cl: int, int_cl: dict = None) -> str:
    if int_cl:
        return f'[CL] Trg: {int_cl[trg_cl]} - Prd: {int_cl[pred_cl]}'
    else:
        return f'[CL] Trg: {trg_cl} - Prd: {pred_cl}'


def switch_key_val_dict(d: dict) -> dict:
    out = dict()
    for k in d:
        assert d[k] not in out, 'more than 1 key with same value. wrong.'
        out[d[k]] = k

    return out


def plot_progress_cams(ds, model, frms_idx: list, outd: str, args,
                       iteration: int, cl_int: dict = None):

    if args.task in [constants.F_CL, constants.TCAM]:
        req_grad = False
    elif args.task == constants.C_BOX:
        req_grad = False
    elif args.task == constants.STD_CL:
        req_grad = constants.METHOD_REQU_GRAD[args.method]
    else:
        raise NotImplementedError

    if args.task == constants.STD_CL:
        cam_extractor = _build_std_cam_extractor(classifier=model, args=args)
    elif args.task == constants.F_CL:
        cam_extractor = _build_fcam_extractor(model=model, args=args)
        # useful for drawing side-by-side.
        # todo: build classifier from scratch and create its cam extractor.
    elif args.task == constants.TCAM:
        cam_extractor = _build_tcam_extractor(model=model, args=args)
    elif args.task == constants.C_BOX:
        cam_extractor = None
    else:
        raise NotImplementedError

    viz = Viz_WSOL()
    int_cl = switch_key_val_dict(cl_int) if cl_int else None

    transformer_copy = copy.deepcopy(ds.transform)
    ds.transform = get_eval_tranforms(args.crop_size)
    DLLogger.log('switched train transform to valid')


    for _image_id in tqdm(frms_idx, ncols=constants.NCOLS, total=len(frms_idx)):
        shot_idx = ds.index_id[ds.frame_to_shot_idx[_image_id]]
        vals = ds._get_one_item(idx=shot_idx, frame_id=_image_id)
        image, target, image_id, raw_img, std_cam, _, _, _ = vals

        image = image.cuda(args.c_cudaid)  # 3, h, w.
        image_size = image.shape[1:]
        # raw_img: 3, h, w
        raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
        raw_img = raw_img.astype(np.uint8)

        with torch.set_grad_enabled(req_grad):
            # todo: this is NOT low res.
            low_cam, cl_logits, cbox = get_cam_one_sample(
                args=args, model=model, cam_extractor=cam_extractor,
                image=image.unsqueeze(0), target=target)

        p_cl = cl_logits.argmax(dim=1).item()

        tag_cl = get_str_trg_prd_cl(pred_cl=p_cl, trg_cl=target, int_cl=int_cl)

        with torch.no_grad():
            cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                image_size,
                                mode='bilinear',
                                align_corners=False
                                ).squeeze(0).squeeze(0)

            cam = torch.clamp(cam, min=0.0, max=1.)
            cam = torch.clamp(cam, min=0.0, max=1.)
            cam = t2n(cam)

            # cams shape (h, w).
            assert cam.shape == image_size

            cam_resized = cam
            cam_normalized = cam_resized
            check_scoremap_validity(cam_normalized)

            if torch.is_tensor(std_cam):
                std_cam = std_cam.squeeze() # h, w
                std_cam = F.interpolate(std_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                std_cam = torch.clamp(std_cam, min=0.0, max=1.)
                std_cam = torch.clamp(std_cam, min=0.0, max=1.)
                std_cam = t2n(std_cam)
                check_scoremap_validity(std_cam)
                assert std_cam.shape == image_size
            else:
                std_cam = None

        # case 1: bbox.
        # todo: fix this. add more datasets with bbox. deal with other types.
        if args.dataset in [constants.YTOV1, constants.YTOV22]:

            if cam_normalized.min() == cam_normalized.max():
                th = 0.
            else:
                th = threshold_otsu(cam_normalized)

            l_bbox, nbr_bbox = compute_bboxes_from_scoremaps_ext_contours(
                scoremap=cam_normalized, scoremap_threshold_list=[th],
                multi_contour_eval=True, bbox=None)

            assert len(l_bbox) == 1
            largest_bbox = get_largest_bbox(bboxes=l_bbox[0])
            if largest_bbox is not None:  # (1, 4). np.ndarary
                assert largest_bbox.shape == (1, 4), largest_bbox.shape

            l_bbox: np.ndarray = l_bbox[0]  # (nbr_bx, 4)

            if l_bbox.size == 0:  # no bbox found ?!
                continue

            assert l_bbox.shape[1] == 4, l_bbox.shape[1]
            assert l_bbox.ndim == 2, l_bbox.ndim

            datum = {'img': raw_img,
                     'img_id': image_id,
                     'gt_bboxes': None,
                     'gt_matched_bbox': None,
                     'pred_bbox': largest_bbox,
                     'bboxes': l_bbox,
                     'iou': None,
                     'tau': None,
                     'sigma': None,
                     'cam': cam_normalized,
                     'std_cam': std_cam,
                     'tag_cl': tag_cl,
                     'iteration': iteration
                     }

            outf = join(outd, format(reformat_id(image_id)))
            os.makedirs(outf, exist_ok=True)
            outf = join(outf, f'{iteration:010}.png')
            viz.plot_single(datum=datum, outf=outf,plot_all_instances=True)

        else:
            raise NotImplementedError

    ds.transform = transformer_copy
    DLLogger.log('switched train transform back to train')


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


def get_cm():
    col_dict = dict()
    for i in range(256):
        col_dict[i] = 'k'

    col_dict[0] = 'k'
    col_dict[int(255 / 2)] = 'y'
    col_dict[255] = 'r'
    colormap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    return colormap


def plot_self_learning(_id: str, raw_img: torch.Tensor,
                       cam: torch.Tensor, roi: torch.Tensor,
                       msk: torch.Tensor, bb: torch.Tensor, fdout: str,
                       iteration: int):
    # raw_img: 3, h, w
    raw_img = raw_img.permute(1, 2, 0).cpu().squeeze().numpy()  # h, w, 3
    raw_img = raw_img.astype(np.uint8)

    _cl = mcolors.CSS4_COLORS['orange']
    lims = [(raw_img, 'IMG'), (cam_2Img(cam), 'CAM')]
    lims.append((roi.squeeze().cpu().numpy().astype(np.uint8), 'ROI'))
    lims.append((msk.squeeze().cpu().numpy().astype(np.uint8), 'bb mask'))
    lims.append((bb.cpu().numpy(), 'bbox'))

    bbo_info = None
    for i, (cnt, tag) in enumerate(lims):
        if tag == 'bbox':
            bbo_info = convert_bbox(cnt)

    nrows = 1
    ncols = len(lims)

    if bbo_info is not None:
        ncols = ncols - 1
        tmp = [el for el in lims if el[1] != 'bbox']
        _lims = tmp

    him, wim = _lims[1][0].shape
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
                                         edgecolor=_cl,
                                         facecolor='none')
            axes[0, i].add_patch(rect_roi)

        axes[0, i].text(3, 40, tag,
                        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8})
    plt.suptitle(f'{_id}')

    # outf = join(fdout, format(reformat_id(_id)))
    os.makedirs(fdout, exist_ok=True)
    outf = join(fdout, f'{iteration}_{format(reformat_id(_id))}.png')

    fig.savefig(outf, pad_inches=0, bbox_inches='tight', dpi=200,
                optimize=True)
    plt.close(fig)

