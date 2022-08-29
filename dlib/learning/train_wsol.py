import os
import sys
import time
import warnings
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple, Dict, Any
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import gpu_memory_stats
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.cams.fcam_seeding import MBSeederSLFCAMS
from dlib.cams.tcam_seeding import TCAMSeeder

from dlib.cams.fcam_seeding import SeederCBOX

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.shared import is_cc
from dlib.datasets.ilsvrc_manager import prepare_next_bucket
from dlib.datasets.ilsvrc_manager import prepare_vl_tst_sets
from dlib.datasets.ilsvrc_manager import delete_train

from dlib.parallel import sync_tensor_across_gpus
from dlib.parallel import MyDDP as DDP

from dlib.utils.utils_checkpoints import save_checkpoint
from dlib.utils.utils_checkpoints import fn_keep_last_n_checkpoints
from dlib.utils.utils_checkpoints import find_last_checkpoint
from dlib.utils.utils_checkpoints import move_state_dict_to_device

from dlib.visualization.vision_progress import plot_progress_cams
from dlib.visualization.vision_progress import plot_self_learning

from dlib.cams import build_tcam_extractor
from dlib.cams.tcam_seeding import GetRoiSingleCam

from dlib.box import BoxStats
from dlib.filtering import GaussianFiltering

from dlib.configure.config import get_root_wsol_dataset

from dlib import losses


__all__ = ['Basic', 'Trainer']


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        val = constants.VALIDSET
        self.value_per_epoch = [] \
            if split == val else [-np.inf if higher_is_better else np.inf]
        # todo: replace with self.value_per_epoch = []

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)
        # todo: change to
        # idx = [i for i, x in enumerate(
        #  self.value_per_epoch) if x == self.best_value]
        # assert len(idx) > 0
        # self.best_epoch = idx[-1]


class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)
    _EVAL_METRICS = ['loss',
                     constants.CLASSIFICATION_MTR,
                     constants.LOCALIZATION_MTR,
                     constants.FAILD_BOXES_MTR]
    _BEST_CRITERION_METRIC = constants.LOCALIZATION_MTR

    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
        constants.YTOV1: constants.NUMBER_CLASSES[constants.YTOV1],
        constants.YTOV22: constants.NUMBER_CLASSES[constants.YTOV22]
    }

    # @property
    # def _BEST_CRITERION_METRIC(self):
    #     assert self.inited
    #     assert self.args is not None
    #
    #     return 'localization'
    #
    #     if self.args.task == constants.STD_CL:
    #         return 'classification'
    #     elif self.args.task == constants.F_CL:
    #         return 'localization'
    #     else:
    #         raise NotImplementedError

    def __init__(self, args):
        self.args = args
        
    def _set_performance_meters(self):
        self._EVAL_METRICS += [f'{constants.LOCALIZATION_MTR}_IOU_{threshold}'
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top1_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top5_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict


def plot_perf_curves_top_1_5(curves: dict, fdout: str, title: str):

    x_label = r'$\tau$'
    y_label = 'BoxAcc'

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

    for i, top in enumerate(['top1', 'top5']):

        iouthres = sorted(list(curves[top].keys()))
        for iout in iouthres:
            axes[0, i].plot(curves['x'], curves[top][iout],
                            label=r'{}: $\sigma$={}'.format(top, iout))

        axes[0, i].xaxis.set_tick_params(labelsize=5)
        axes[0, i].yaxis.set_tick_params(labelsize=5)
        axes[0, i].set_xlabel(x_label, fontsize=8)
        axes[0, i].set_ylabel(y_label, fontsize=8)
        axes[0, i].grid(True)
        axes[0, i].legend(loc='best')
        axes[0, i].set_title(top)

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                dpi=300)


def plot_perf_curves_top_1_5(curves: dict, fdout: str, title: str):

    x_label = r'$\tau$'
    y_label = 'BoxAcc'

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

    for i, top in enumerate(['top1', 'top5']):

        iouthres = sorted(list(curves[top].keys()))
        for iout in iouthres:
            axes[0, i].plot(curves['x'], curves[top][iout],
                            label=r'{}: $\sigma$={}'.format(top, iout))

        axes[0, i].xaxis.set_tick_params(labelsize=5)
        axes[0, i].yaxis.set_tick_params(labelsize=5)
        axes[0, i].set_xlabel(x_label, fontsize=8)
        axes[0, i].set_ylabel(y_label, fontsize=8)
        axes[0, i].grid(True)
        axes[0, i].legend(loc='best')
        axes[0, i].set_title(top)

    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                dpi=300)


class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss: losses.MasterLoss,
                 classifier=None,
                 current_step: int = 0):
        super(Trainer, self).__init__(args=args)

        self.device = torch.device(args.c_cudaid)
        self.args = args
        self.performance_meters = self._set_performance_meters()
        self.update_performance_tracker_from_checkpoint()

        self.model = model
        self.current_step = current_step
        self.checkpoints_health = dict()
        self.cpt_trackers_health = dict()
        self.cpt_best_models_health = {
            constants.BEST_CL: dict(),
            constants.BEST_LOC: dict()
        }

        if isinstance(model, DDP):
            self._pytorch_model = self.model.module
        else:
            self._pytorch_model = self.model

        self.loss: losses.MasterLoss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if is_cc() and self.args.ds_chunkable:
            if self.args.distributed:
                dist.barrier()
            if self.args.is_node_master:
                status1 = prepare_vl_tst_sets(dataset=self.args.dataset)
                if (status1[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing valid/test. '
                                 f'{status1[1]}. Exiting.')

                status2 = prepare_next_bucket(bucket=0,
                                              dataset=self.args.dataset)
                if (status2[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing bucket '
                                 f'{0}. {status2[1]}. Exiting.')

                if (status1[0] == -1) or (status2[0] == -1):
                    sys.exit()
            if self.args.distributed:
                dist.barrier()

        self.loaders, self.train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            dataset=self.args.dataset,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=0
        )

        self.sl_mask_builder = None
        if args.task == constants.F_CL:
            self.sl_mask_builder: MBSeederSLFCAMS = self._get_sl(args)
        if args.task == constants.TCAM:
            self.sl_mask_builder: TCAMSeeder = self._get_sl(args)

        self.fg_mask_seed_builder = None
        self.bg_mask_seed_builder = None
        self.kde = None

        if args.task == constants.C_BOX:
            self.mask_seed_builder: SeederCBOX = self._get_cbox_seeder(args)

        self.epoch = 0
        self.counter = 0
        self.seed = int(args.MYSEED)
        self.default_seed = int(args.MYSEED)

        # todo: weak. best models should be loaded from best models when
        #  using checkpoints.
        self.best_model_loc = deepcopy(self._pytorch_model).to(
            self.device).eval()
        self.best_model_cl = deepcopy(self._pytorch_model).to(
            self.device).eval()

        self.load_checkpoint(checkpoint_type=constants.BEST_LOC,
                             net=self.best_model_loc)
        self.load_checkpoint(checkpoint_type=constants.BEST_CL,
                             net=self.best_model_cl)

        self.perf_meters_backup = None
        self.inited = True

        self.classifier = classifier
        self.box_stats = None
        self.blur_op = None
        if args.task == constants.C_BOX:
            assert classifier is not None
            self.classifier.eval()
            self.classifier.freeze_classifier()
            self.classifier.assert_cl_is_frozen()

            self.box_stats = BoxStats(scale_domain=args.model['scale_domain'],
                                      h=args.crop_size,
                                      w=args.crop_size).cuda(args.c_cudaid)

            self.blur_op: GaussianFiltering = GaussianFiltering(
                blur_ksize=args.cb_cl_score_blur_ksize,
                blur_sigma=args.cb_cl_score_blur_sigma,
                device=torch.device(self.args.c_cudaid)).cuda(args.c_cudaid)

        self.std_cam_extractor = None
        if args.task in [constants.F_CL, constants.TCAM]:
            assert classifier is not None
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.tcam_extractor = None
        if args.task == constants.TCAM:
            self.tcam_extractor = self._build_tcam_extractor(
                model=self._pytorch_model, args=self.args)

        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.vl_size_priors = None
        if self._is_prior_size_needed():
            self.vl_size_priors: dict = deepcopy(
                self.loaders[constants.VALIDSET].dataset.build_size_priors())

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()

        self.pre_forward_info = dict()
        self.stats = dict()

        self.selected_frm_idx_tr_cam_progress = None

    @staticmethod
    def _build_tcam_extractor(model, args):
        return build_tcam_extractor(model=model, args=args)

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    @staticmethod
    def _get_sl(args):
        if args.task == constants.F_CL:
            return MBSeederSLFCAMS(
                    min_=args.sl_min,
                    max_=args.sl_max,
                    ksz=args.sl_ksz,
                    min_p=args.sl_min_p,
                    fg_erode_k=args.sl_fg_erode_k,
                    fg_erode_iter=args.sl_fg_erode_iter,
                    support_background=args.model['support_background'],
                    multi_label_flag=args.multi_label_flag,
                    seg_ignore_idx=args.seg_ignore_idx)

        elif args.task == constants.TCAM:
            return TCAMSeeder(
                seed_tech=args.sl_tc_seed_tech,
                min_=args.sl_tc_min,
                max_=args.sl_tc_max,
                ksz=args.sl_tc_ksz,
                max_p=args.sl_tc_max_p,
                min_p=args.sl_tc_min_p,
                fg_erode_k=args.sl_tc_fg_erode_k,
                fg_erode_iter=args.sl_tc_fg_erode_iter,
                support_background=args.model['support_background'],
                multi_label_flag=args.multi_label_flag,
                seg_ignore_idx=args.seg_ignore_idx,
                cuda_id=args.c_cudaid,
                roi_method=args.sl_tc_roi_method,
                p_min_area_roi=args.sl_tc_roi_min_size,
                use_roi=args.sl_tc_use_roi
            )

        else:
            raise NotImplementedError(args.task)

    def _get_cbox_seeder(self, args):
        return SeederCBOX(n=args.cb_seed_n,
                          bg_low_z=args.cb_seed_bg_low_z,
                          bg_up_z=args.cb_seed_bg_up_z,
                          fg_erode_k=args.cb_seed_erode_k,
                          fg_erode_iter=args.cb_seed_erode_iter,
                          ksz=args.cb_seed_ksz,
                          seg_ignore_idx=args.seg_ignore_idx,
                          device=self.device
                          )

    def prepare_std_cams_disq(self, std_cams: torch.Tensor,
                              image_size: Tuple) -> torch.Tensor:

        assert std_cams.ndim == 4
        cams = std_cams.detach()

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        # Quick fix: todo...
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        return cams

    def get_std_cams_minibatch(self, images, targets) -> torch.Tensor:
        # used only for task f_cl
        assert self.args.task == constants.F_CL
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            cl_logits = self.classifier(image.unsqueeze(0))
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def is_seed_required(self, _epoch: int) -> bool:
        if self.args.task == constants.F_CL:
            cmd = (self.args.task == constants.F_CL)
            cmd &= self.args.sl_fc
            cmd &= ('self_learning_fcams' in self.loss.n_holder)
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SelfLearningFcams):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        elif self.args.task == constants.TCAM:
            cmd = (self.args.task == constants.TCAM)
            cmd &= self.args.sl_tc
            cmd &= ('self_learning_tcams' in self.loss.n_holder)
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SelfLearningTcams):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        elif self.args.task == constants.C_BOX:
            cmd = (self.args.task == constants.C_BOX)
            cmd &= ('seed_cbox' in self.loss.n_holder)
            cmd &= self.args.cb_seed
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.SeedCbox):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        return False

    def is_compute_stats_pred_required(self, _epoch: int) -> bool:
        if self.args.task == constants.TCAM:
            cmd = (self.args.task == constants.TCAM)
            cmd &= self.args.sizefg_tmp_tc
            cmd &= ('fg_size_tcams' in self.loss.n_holder)
            cmd2 = False
            for _l in self.loss.losses:
                if isinstance(_l, losses.FgSizeTcams):
                    cmd2 = _l.is_on(_epoch=_epoch)

            return cmd and cmd2

        return False

    def _is_prior_size_needed(self) -> bool:
        cmd = (self.args.task == constants.C_BOX)
        cmd &= (self.args.dataset in [constants.CUB,
                                      constants.ILSVRC,
                                      constants.YTOV1,
                                      constants.YTOV22])
        return cmd

    def _gen_rand_init_box(self, h: int, w: int, minsz: float):
        assert 0 < self.args.cb_init_box_size <= 1.
        assert isinstance(self.args.cb_init_box_size, float)

        assert isinstance(self.args.cb_init_box_var, float)
        assert self.args.cb_init_box_var >= 0

        maxsz = 0.99

        m = self.args.cb_init_box_size
        v = self.args.cb_init_box_var

        s = np.random.normal(loc=m, scale=v, size=(1,)).item()
        s = min(max(s, minsz), maxsz)

        x_hat_0 = max(h / 2. - h * np.sqrt(s) / 2., .0)
        x_hat_1 = min(h / 2. + h * np.sqrt(s) / 2., h - 1)

        y_hat_0 = max(w / 2. - w * np.sqrt(s) / 2., .0)
        y_hat_1 = min(w / 2. + w * np.sqrt(s) / 2., w - 1)
        return x_hat_0, x_hat_1, y_hat_0, y_hat_1

    def _cbox_filter_valid_tensors(self, tensor: torch.Tensor,
                                   valid: torch.Tensor
                                   ) -> Union[torch.Tensor, None]:

        idx_valid = torch.nonzero(valid.contiguous().view(-1, ),
                                  as_tuple=False).squeeze()
        if idx_valid.numel() == 0:
            return None

        _z = tensor[idx_valid]  # ?
        if idx_valid.numel() == 1:
            _z = _z.unsqueeze(0)

        return _z

    def _visualize_cams_train_progress(self, loader, n: int = 200):
        set_seed(seed=self.default_seed, verbose=False)

        if self.args.dataset not in [constants.YTOV1, constants.YTOV22]:
            raise NotImplementedError  # todo: deal with image ids.

        if self.selected_frm_idx_tr_cam_progress is None:
            # ytov1, constants.YTOV22
            shots_ids: list = loader.dataset.image_ids
            total_s = len(shots_ids)
            nbr = min(n, total_s)
            idx = np.random.choice(a=total_s, size=nbr, replace=False).flatten()
            selected_shots_ids = [shots_ids[z] for z in idx]

            selected_frm_idx = []
            for idx in selected_shots_ids:
                l_frames = loader.dataset.index_of_frames[idx]
                fr_idx = np.random.randint(low=0, high=len(l_frames),
                                           size=1).item()
                selected_frm_idx.append(l_frames[fr_idx])

            self.selected_frm_idx_tr_cam_progress = selected_frm_idx
        else:
            selected_frm_idx = self.selected_frm_idx_tr_cam_progress

        outd = join(self.args.outd_backup, 'debug', constants.TRAINSET,
                    'progress-vizu')
        DLLogger.log(f'Visualizing cams progress of {n} samples...')
        plot_progress_cams(ds=loader.dataset, model=self._pytorch_model,
                           frms_idx=selected_frm_idx, outd=outd,
                           args=self.args, iteration=self.epoch)

        set_seed(seed=self.default_seed, verbose=False)

    def _pre_forward(self, output, images: torch.Tensor,
                     vl_size_priors: Dict[str, Any], images_id: list):

        n, c, h, w = images.shape
        ratio = float(h * w)

        if self.args.task == constants.STD_CL:
            pass

        elif self.args.task == constants.F_CL:
            pass

        elif self.args.task == constants.TCAM:


            if self.is_compute_stats_pred_required(self.epoch):
                split = constants.TRAINSET
                if split not in self.stats:
                    self.stats[split] = dict()

                ksz = constants.KEY_CAM_FG_SZ
                if ksz not in self.stats[split]:
                    self.stats[split][ksz] = dict()

                _raw_cams = deepcopy(self.tcam_extractor.model.cams)
                area = []
                total = 0.0

                for i in range(_raw_cams.shape[0]):

                    self.tcam_extractor.model.cams = _raw_cams[
                        i].unsqueeze(0)
                    _built_cam = self.tcam_extractor().detach()  # h, w
                    # stats.
                    total = _built_cam.shape[0] * _built_cam.shape[1]
                    area.append(_built_cam.sum())

                area = torch.tensor(area, device=_raw_cams.device,
                                    requires_grad=False) / float(total)
                area = area.cpu()
                for i in range(_raw_cams.shape[0]):
                    _id = images_id[i]
                    _sz = area[i].item()
                    self.stats[split][ksz][_id] = _sz

        elif self.args.task == constants.C_BOX:

            # todo: clean later.

            box = output
            _box = box.detach()
            zz = self.box_stats(box=_box, eval=True)
            x_hat, y_hat, valid, area, mask_fg, mask_bg = zz

            _area = area / ratio

            # imgs_fg = self.get_fg_imgs(images=images, blured_imgs=blured_imgs,
            #                            mask_fg=mask_fg, mask_bg=mask_bg)
            # imgs_bg = self.get_bg_imgs(images=images, blured_imgs=blured_imgs,
            #                            mask_fg=mask_fg, mask_bg=mask_bg)

            # logits_fg = self.classifier(imgs_fg)
            # logits_bg = self.classifier(imgs_bg)
            # logits_clean = self.classifier(images)

            # raw_imgs_device = raw_imgs.cuda(self.args.c_cudaid)

            self.pre_forward_info['x_hat'] = x_hat.detach()
            self.pre_forward_info['y_hat'] = y_hat.detach()

            c_data = constants.MIN_SIZE_DATA
            c_cont = constants.MIN_SIZE_CONST
            for i in range(n):

                if self.args.cb_pp_box_min_size_type == c_cont:
                    minsz = self.args.cb_pp_box_min_size
                elif self.args.cb_pp_box_min_size_type == c_data:
                    minsz = vl_size_priors['min_s'][i]
                else:
                    raise NotImplementedError

                if (valid[i] == 0) or (_area[i] < minsz):
                    z = self._gen_rand_init_box(h, w, minsz)
                    self.pre_forward_info['x_hat'][i][0] = z[0]
                    self.pre_forward_info['x_hat'][i][1] = z[1]
                    self.pre_forward_info['y_hat'][i][0] = z[2]
                    self.pre_forward_info['y_hat'][i][1] = z[3]
        else:
            raise NotImplementedError

    def _wsol_training(self,
                       images_id: list,
                       images: torch.Tensor,
                       raw_imgs: torch.Tensor,
                       targets: torch.Tensor,
                       std_cams: torch.Tensor,
                       blured_imgs: torch.Tensor,
                       seq_iter: torch.Tensor,
                       frm_iter: torch.Tensor,
                       vl_size_priors: Dict[str, Any],
                       iteration: int,
                       batch_idx: int,
                       roi: torch.Tensor = None,
                       split: str = constants.TRAINSET,
                       msk_bbox: torch.Tensor = None):
        assert split == constants.TRAINSET, split
        y_global = targets

        output = self.model(images)

        with torch.no_grad():
            self._pre_forward(output=output, images=images,
                              vl_size_priors=vl_size_priors,
                              images_id=images_id)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            loss = self.loss(epoch=self.epoch, cl_logits=cl_logits,
                             glabel=y_global)
            logits = cl_logits

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits

        elif self.args.task == constants.TCAM:
            cl_logits, fcams, im_recon = output

            self_lr = False
            fg_size = None

            if self.is_compute_stats_pred_required(self.epoch):
                if split not in self.stats:
                    self.stats[split] = dict()

                ksz = constants.KEY_CAM_FG_SZ
                if ksz not in self.stats[split]:
                    self.stats[split][ksz] = dict()

            if self.is_seed_required(_epoch=self.epoch):

                tt = self.args.sl_tc_epoch_switch_to_sl
                _cnd = (tt != -1)
                _cnd &= (self.epoch >= tt)
                t2 = self.args.empty_out_bb_tc_start_ep
                _cnd |= self.args.empty_out_bb_tc and (self.epoch >= t2)

                self_lr = _cnd

                if _cnd:  # todo: stop loading classifier's cams in dataset.
                    with torch.no_grad():  # todo: check grad/no grad.
                        self.best_model_loc.eval()
                        tcam_extractor = self._build_tcam_extractor(
                            model=self.best_model_loc, args=self.args)
                        _out_best_model = self.best_model_loc(images)

                    _raw_cams = deepcopy(tcam_extractor.model.cams)
                    cams_inter = None
                    for  i in range(_raw_cams.shape[0]):

                        tcam_extractor.model.cams = _raw_cams[i].unsqueeze(0)
                        _built_cam = tcam_extractor().detach()  # h, w
                        _built_cam = _built_cam.unsqueeze(0).unsqueeze(0)

                        if cams_inter is None:
                            cams_inter = _built_cam
                        else:
                            cams_inter = torch.cat((cams_inter, _built_cam),
                                                   dim=0)

                    # overwrite roi.
                    _roi = None
                    _msk_bbox = None
                    _getter_roi = GetRoiSingleCam(
                        roi_method=constants.ROI_LARGEST,  # todo: add to conf
                        p_min_area_roi=self.args.sl_tc_roi_min_size
                    )
                    _cams_cpu = cams_inter.detach().cpu()
                    ksz = constants.KEY_CAM_FG_SZ
                    fg_size = []

                    for k in range(cams_inter.shape[0]):

                        s_roi, s_msk, s_bb = _getter_roi(
                            _cams_cpu[k].squeeze(), thresh=None)
                        s_roi = s_roi.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w
                        s_msk = s_msk.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w
                        _, _, h, w = s_roi.shape
                        # debug plot.
                        if batch_idx == 0:
                            _outd = join(self.args.outd_backup, 'debug',
                                         constants.TRAINSET,
                                         'progress-roi', f'{self.epoch}')
                            # plot_self_learning(
                            #     _id=images_id[k],
                            #     raw_img=raw_imgs[k],
                            #     cam=_cams_cpu[k].squeeze(),
                            #     roi=s_roi, msk=s_msk, bb=s_bb, fdout=_outd,
                            #     iteration=self.epoch
                            # )
                        # todo: stats.
                        if self.is_compute_stats_pred_required(self.epoch):
                            _id = images_id[k]
                            _tmp = _cams_cpu[k].unsqueeze(0).unsqueeze(
                                0) * s_roi
                            sz = _tmp.sum() / (h * w)  # cpu.
                            sz = sz.item()
                            self.stats[split][ksz][_id] = sz
                            fg_size.append(sz)

                        if _roi is None:
                            _roi = s_roi  # 1, 1, h, w
                            _msk_bbox = s_msk
                        else:
                            _roi = torch.cat((_roi, s_roi), dim=0)
                            _msk_bbox = torch.cat((_msk_bbox, s_msk), dim=0)

                    _roi = _roi.cuda(self.args.c_cudaid)
                    _msk_bbox = _msk_bbox.cuda(self.args.c_cudaid)
                    msk_bbox = _msk_bbox  # b, 1, h, w
                    if fg_size is not None:
                        fg_size = torch.tensor(fg_size, dtype=torch.float,
                                               requires_grad=False,
                                               device=self.device)

                else:

                    if std_cams is None:
                        cams_inter = self.get_std_cams_minibatch(
                            images=images, targets=targets)
                    else:
                        cams_inter = std_cams

                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(
                        images=images, targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(x=cams_inter, roi=roi)

                    # if self_lr:
                    #     seeds = self.sl_mask_builder.use_all_roi(x=cams_inter,
                    #                                              roi=roi)
                    # else:
                    #     seeds = self.sl_mask_builder(x=cams_inter, roi=roi)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds,
                seq_iter=seq_iter,
                frm_iter=frm_iter,
                fg_size=fg_size,
                msk_bbox=msk_bbox
            )
            logits = cl_logits

        elif self.args.task == constants.C_BOX:
            box = output
            zz = self.box_stats(box=box, eval=False)
            x_hat, y_hat, valid, area, mask_fg, mask_bg = zz

            logits_fg = None
            logits_bg = None
            logits_clean = None

            imgs_fg = self.get_fg_imgs(images=images,
                                       blured_imgs=blured_imgs,
                                       mask_fg=mask_fg, mask_bg=mask_bg)
            logits_fg = self.classifier(imgs_fg)

            if self.args.cb_cl_score:
                imgs_bg = self.get_bg_imgs(images=images,
                                           blured_imgs=blured_imgs,
                                           mask_fg=mask_fg, mask_bg=mask_bg)
                logits_bg = self.classifier(imgs_bg)
                logits_clean = self.classifier(images)


            cams_inter, seeds = None, None
            if self.is_seed_required(_epoch=self.epoch):
                assert std_cams is not None
                cams_inter = std_cams
                with torch.no_grad():
                    seeds = self.mask_seed_builder(cams_inter)

                # seeds = self._cbox_filter_valid_tensors(seeds, valid)

            loss = self.loss(
                epoch=self.epoch,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                seeds=seeds,
                raw_scores=box,
                x_hat=x_hat,
                y_hat=y_hat,
                valid=valid,
                area=area,
                mask_fg=mask_fg,
                mask_bg=mask_bg,
                logits_fg=logits_fg,
                logits_bg=logits_bg,
                logits_clean=logits_clean,
                pre_x_hat=self.pre_forward_info['x_hat'],
                pre_y_hat=self.pre_forward_info['y_hat'],
                vl_size_priors=vl_size_priors
            )

            logits = logits_fg
        else:
            raise NotImplementedError

        return logits, loss

    def on_epoch_start(self):
        torch.cuda.empty_cache()

        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.model.train(mode=True)

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

        for item in self.loaders:
            self.loaders[item].dataset.set_epoch(self.epoch)
            self.loaders[item].dataset.set_epoch_tmp_manager()

        if self.args.task == constants.TCAM:
            seed_tech = self.loaders[
                constants.TRAINSET].dataset.tmp_manager.sl_tc_seed_tech
            self.sl_mask_builder.set_seed_tech(seed_tech)
            msg = self.loaders[
                constants.TRAINSET].dataset.tmp_manager.get_current_status()
            DLLogger.log(f'tmp-manager: {msg}')

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))

        torch.cuda.empty_cache()

    def random(self):
        self.counter = self.counter + 1
        self.seed = self.seed + self.counter
        set_seed(seed=self.seed, verbose=False)

    def reload_data_bucket(self, tr_bucket: int):

        loaders, train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            dataset=self.args.dataset,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=tr_bucket
        )

        self.train_sampler = train_sampler
        self.loaders[constants.TRAINSET] = loaders[constants.TRAINSET]

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch + tr_bucket)

    @staticmethod
    def _fill_minibatch(_x: torch.Tensor, mbatchsz: int) -> torch.Tensor:
        assert isinstance(_x, torch.Tensor)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0
        assert _x.shape[0] <= mbatchsz, f'{_x.shape[0]}, {mbatchsz}'

        if _x.shape[0] == mbatchsz:
            return _x

        s = _x.shape[0]
        t = math.ceil(float(mbatchsz) / s)
        v = torch.cat(t * [_x])
        assert v.shape[1:] == _x.shape[1:]

        out = v[:mbatchsz]
        assert out.shape[0] == mbatchsz
        return out

    @staticmethod
    def _fill_minibatch_list(_x: list, mbatchsz: int) -> list:
        assert isinstance(_x, list)
        assert isinstance(mbatchsz, int)
        assert mbatchsz > 0
        assert len(_x) <= mbatchsz, f'{len(_x)}, {mbatchsz}'

        if len(_x) == mbatchsz:
            return _x

        s = len(_x)
        t = math.ceil(float(mbatchsz) / s)
        v = t * _x

        out = v[:mbatchsz]
        assert len(out) == mbatchsz
        return out

    def train(self, split, epoch):
        assert split == constants.TRAINSET

        self.epoch = epoch
        # progress visu.
        _cnd = (self.args.is_master and self.args.plot_tr_cam_progress)
        _cnd &= (self.args.plot_tr_cam_progress_n > 0)
        if _cnd:
            _n  =self.args.plot_tr_cam_progress_n
            assert _n > 0, _n
            assert isinstance(_n, int), type(_n)
            self._visualize_cams_train_progress(loader=self.loaders[split],
                                                n=_n)
            torch.cuda.empty_cache()

        if self.args.distributed:
            dist.barrier()

        self.random()
        self.on_epoch_start()

        nbr_tr_bucket = self.args.nbr_buckets
        if not self.args.ds_chunkable:
            nbr_tr_bucket = 1

        loader = self.loaders[split]

        total_loss = None
        num_correct = 0
        num_images = 0

        mbatchsz = self.args.batch_size_backup

        distributed = self.args.distributed
        scaler = GradScaler(enabled=self.args.amp)

        iteration = 0

        for bucket in range(nbr_tr_bucket):

            status = 0
            if self.args.ds_chunkable:
                if is_cc():
                    ddp_barrier(distributed)
                    if self.args.is_node_master:
                        if bucket > 0:
                            delete_train(bucket=bucket - 1,
                                         dataset=self.args.dataset)

                        status = prepare_next_bucket(bucket=bucket,
                                                     dataset=self.args.dataset)
                        if (status == -1) and self.args.is_master:
                            DLLogger.log(f'Error in preparing bucket '
                                         f'{bucket}. Exiting.')

                ddp_barrier(distributed)

                if status == -1:
                    sys.exit()
                self.reload_data_bucket(tr_bucket=bucket)
                loader = self.loaders[split]

            ddp_barrier(distributed)

            for batch_idx, (
                    images, targets, images_id, raw_imgs, std_cams,
                    seq_iter, frm_iter, roi) in tqdm(
                    enumerate(loader), ncols=constants.NCOLS,
                    total=len(loader), desc=f'BUCKET {bucket}/{nbr_tr_bucket}'):

                self.current_step += 1

                self.random()

                vl_size_priors: dict = None
                if self._is_prior_size_needed():
                    vl_size_priors: Dict[str, Any] = \
                        self._build_mbatch_size_prior(targets)
                    for kz in vl_size_priors:
                        vl_size_priors[kz] = self._fill_minibatch(
                            vl_size_priors[kz], mbatchsz).cuda(
                            self.args.c_cudaid)

                images = self._fill_minibatch(images, mbatchsz)
                targets = self._fill_minibatch(targets, mbatchsz)
                raw_imgs = self._fill_minibatch(raw_imgs, mbatchsz)
                # images_id: tuple.
                images_id: list = list(images_id)
                images_id = self._fill_minibatch_list(images_id, mbatchsz)
                seq_iter = self._fill_minibatch(seq_iter, mbatchsz)
                frm_iter = self._fill_minibatch(frm_iter, mbatchsz)

                images = images.cuda(self.args.c_cudaid)
                targets = targets.cuda(self.args.c_cudaid)

                blured_imgs = None
                if self.args.task == constants.C_BOX:
                    blured_imgs = self.blur_op(images=images)

                if roi.ndim == 1:
                    roi = None
                else:
                    roi = self._fill_minibatch(roi, mbatchsz)
                    roi = roi.cuda(self.args.c_cudaid)

                if std_cams.ndim == 1:
                    std_cams = None
                else:
                    assert std_cams.ndim == 4
                    std_cams = self._fill_minibatch(std_cams, mbatchsz)
                    std_cams = std_cams.cuda(self.args.c_cudaid)

                    with autocast(enabled=self.args.amp):
                        with torch.no_grad():
                            std_cams = self.prepare_std_cams_disq(
                                std_cams=std_cams, image_size=images.shape[2:])

                self.optimizer.zero_grad(set_to_none=True)

                # with torch.autograd.set_detect_anomaly(True):
                with autocast(enabled=self.args.amp):
                    logits, loss = self._wsol_training(
                        images_id, images, raw_imgs, targets,
                        std_cams, blured_imgs, seq_iter, frm_iter,
                        vl_size_priors, iteration, batch_idx,
                        split=constants.TRAINSET, roi=roi)

                with torch.no_grad():
                    pred = logits.argmax(dim=1).detach()

                    if total_loss is None:
                        total_loss = loss.detach().squeeze() * images.size(0)
                    else:
                        total_loss += loss.detach().squeeze() * images.size(0)

                    num_correct += (pred == targets).sum().detach()
                    num_images += images.shape[0]

                if loss.requires_grad and torch.isfinite(loss).item():
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                ddp_barrier(distributed)
                cnd = (self.current_step % self.args.checkpoint_save == 0)
                cnd &= self.args.is_master

                if cnd:
                    # store in scratch in case of cc.
                    _save_dir = join(self.args.outd_backup,
                                     self.args.save_dir_models)
                    save_checkpoint(
                        network=self._pytorch_model,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        loss=self.loss,
                        save_dir=_save_dir,
                        current_step=self.current_step,
                        key=constants.CHP_CP
                    )

                    fn_keep_last_n_checkpoints(
                        _save_dir, n=self.args.keep_last_n_checkpoints,
                        checkpoints_health=self.checkpoints_health,
                        key=constants.CHP_CP
                    )

                    self.save_checkpoint_performance_tracker()

                iteration += 1

        if self.args.distributed:
            num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
            nxx = torch.tensor([num_images], dtype=torch.float,
                               requires_grad=False, device=torch.device(
                    self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nxx).sum().item()
            total_loss = sync_tensor_across_gpus(total_loss.view(1, )).sum()
            dist.barrier()

        loss_average = total_loss.item() / float(num_images)
        classification_acc = num_correct.item() / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def _build_mbatch_size_prior(self,
                                 glabels: torch.Tensor) -> Dict[str, Any]:
        assert self.vl_size_priors is not None
        assert glabels.ndim == 1

        out: Dict[str, Any] = dict()
        k_labels = list(self.vl_size_priors.keys())
        for k in self.vl_size_priors[k_labels[0]]:
            out[k] = torch.zeros_like(glabels, dtype=torch.float32,
                                      requires_grad=False)

        gl: np.ndarray = glabels.cpu().numpy()
        for i in range(gl.size):
            label = gl[i]
            for k in out:
                out[k][i] = self.vl_size_priors[label][k]

        return out

    def print_performances(self, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def save_checkpoint_performance_tracker(self):
        save_dir = join(self.args.outd_backup, self.args.save_dir_models)
        key = constants.CHP_TR
        save_path = os.path.join(save_dir, f'{self.current_step}_{key}.pth')
        torch.save({constants.CHP_TR: self.serialize_perf_meter()}, f=save_path)
        DLLogger.log(f'Saved tracker checkpoint @ {save_path}.')

        fn_keep_last_n_checkpoints(
            save_dir, n=self.args.keep_last_n_checkpoints,
            checkpoints_health=self.cpt_trackers_health,
            key=constants.CHP_TR
        )

    def load_checkpoint_performance_tracker(self) -> dict:
        # write in scratch.
        save_dir = join(self.args.outd_backup, self.args.save_dir_models)
        _, cpt = find_last_checkpoint(save_dir, constants.CHP_TR)
        return cpt[constants.CHP_TR]

    def update_performance_tracker_from_checkpoint(self):
        cpt = self.load_checkpoint_performance_tracker()
        if cpt is None:
            return 0

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                v = cpt[split][metric]['current_value']
                self.performance_meters[split][metric].current_value = v

                v = cpt[split][metric]['best_value']
                self.performance_meters[split][metric].best_value = v

                v = cpt[split][metric]['best_epoch']
                self.performance_meters[split][metric].best_epoch = v

                v = deepcopy(cpt[split][metric]['value_per_epoch'])
                self.performance_meters[split][metric].value_per_epoch = v

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def save_performances(self, epoch=None, checkpoint_type=None):
        tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))
        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            for split in self._SPLITS:
                for metric in self._EVAL_METRICS:

                    f.write("REPORT EPOCH/{}: split: {}/metric {}: {} \n"
                            "".format(epoch, split, metric,
                                      self.performance_meters[split][
                                          metric].current_value))
                    f.write(
                        "REPORT EPOCH/{}: split: {}/metric {}: {}_best "
                        "\n".format(epoch, split, metric,
                                    self.performance_meters[split][
                                        metric].best_value))
    @staticmethod
    def get_fg_imgs(images: torch.Tensor, blured_imgs: torch.Tensor,
                    mask_fg: torch.Tensor, mask_bg: torch.Tensor):
        assert images.ndim == 4
        assert mask_fg.shape[0] == images.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == images.shape[2:]
        assert mask_fg.shape == mask_bg.shape

        return mask_fg * images + mask_bg * blured_imgs

    @staticmethod
    def get_bg_imgs(images: torch.Tensor, blured_imgs: torch.Tensor,
                    mask_fg: torch.Tensor, mask_bg: torch.Tensor):
        assert images.ndim == 4
        assert mask_fg.shape[0] == images.shape[0]
        assert mask_fg.shape[1] == 1
        assert mask_fg.shape[2:] == images.shape[2:]
        assert mask_fg.shape == mask_bg.shape

        return mask_bg * images + mask_fg * blured_imgs

    def cl_forward(self, images: torch.Tensor, blured_imgs: torch.Tensor=None):
        output = self.model(images)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            cl_logits, fcams, im_recon = output

        elif self.args.task == constants.C_BOX:
            box = output
            _, _, valid, _, mask_fg, mask_bg = self.box_stats(box=box,
                                                              eval=True)
            imgs_fg = self.get_fg_imgs(images=images, blured_imgs=blured_imgs,
                                       mask_fg=mask_fg, mask_bg=mask_bg)
            cl_logits = self.classifier(imgs_fg)
        else:
            raise NotImplementedError

        return cl_logits

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids, _, _, _, _, _) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            with torch.no_grad():
                blured_imgs = None
                with autocast(enabled=self.args.amp_eval):
                    if self.args.task == constants.C_BOX:
                        blured_imgs = self.blur_op(images=images)

                    cl_logits = self.cl_forward(images=images,
                                                blured_imgs=blured_imgs
                                                ).detach()

                pred = cl_logits.argmax(dim=1)
                num_correct += (pred == targets).sum().detach()
                num_images += images.size(0)

        # sync
        if self.args.distributed:
            num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
            nx = torch.tensor([num_images], dtype=torch.float,
                              requires_grad=False, device=torch.device(
                    self.args.c_cudaid)).view(1, )
            num_images = sync_tensor_across_gpus(nx).sum().item()
            dist.barrier()

        classification_acc = num_correct / float(num_images) * 100
        if self.args.distributed:
            dist.barrier()

        torch.cuda.empty_cache()
        return classification_acc.item()

    def evaluate(self, epoch, split, checkpoint_type=None, fcam_argmax=False):
        torch.cuda.empty_cache()

        if fcam_argmax:
            assert self.args.task in [constants.F_CL, constants.TCAM]

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task in [constants.F_CL, constants.TCAM]:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        DLLogger.log(fmsg("Evaluate: Epoch {} Split {} {}".format(
            epoch, split, tagargmax)))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
            if not os.path.isdir(outd):
                os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model.eval()
        self._pytorch_model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split][constants.CLASSIFICATION_MTR].update(
            accuracy)

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.CUB,
                                      constants.ILSVRC,
                                      constants.YTOV1,
                                      constants.YTOV22]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self._pytorch_model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=split,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax,
            classifier=self.classifier,
            box_stats=self.box_stats,
            blur_op=self.blur_op
        )

        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg("CAM EVALUATE TIME of {} split: {}".format(
            split, dt.datetime.now() - t0)))

        if self.args.task == constants.C_BOX:
            failed_bbox = cam_computer.get_failed_boxes_mtr()
            self.performance_meters[split][
                constants.FAILD_BOXES_MTR].update(failed_bbox)

        if split == constants.TESTSET and self.args.is_master:
            cam_computer.draw_some_best_pred(rename_ordered=True)

        if self.args.multi_iou_eval or (self.args.dataset ==
                                        constants.OpenImages):
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split][constants.LOCALIZATION_MTR].update(
            loc_score)

        if self.args.dataset in [constants.CUB,
                                 constants.ILSVRC,
                                 constants.YTOV1,
                                 constants.YTOV22]:
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    f'{constants.LOCALIZATION_MTR}_IOU_{IOU_THRESHOLD}'].update(
                    cam_performance[idx])

                self.performance_meters[split][
                    'top1_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top1[idx])

                self.performance_meters[split][
                    'top5_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top5[idx])

            if split == constants.TESTSET and self.args.is_master:
                curve_top_1_5 = cam_computer.evaluator.curve_top_1_5
                with open(join(outd, 'curves_top_1_5.pkl'), 'wb') as fc:
                    pkl.dump(curve_top_1_5, fc, protocol=pkl.HIGHEST_PROTOCOL)

                title = get_tag(self.args, checkpoint_type=checkpoint_type)
                title = 'Top1/5: {}'.format(title)

                if fcam_argmax:
                    title += '_argmax_true'
                else:
                    title += '_argmax_false'
                plot_perf_curves_top_1_5(curves=curve_top_1_5, fdout=outd,
                                              title=title)

        if split == constants.TESTSET and self.args.is_master:

            curves = cam_computer.evaluator.curve_s
            with open(join(outd, 'curves.pkl'), 'wb') as fc:
                pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            title = get_tag(self.args, checkpoint_type=checkpoint_type)

            if fcam_argmax:
                title += '_argmax_true'
            else:
                title += '_argmax_false'
            self.plot_perf_curves(curves=curves, fdout=outd, title=title)

            with open(join(outd, f'thresholds-{checkpoint_type}.yaml'),
                      'w') as fth:
                yaml.dump({
                    'iou_threshold_list':
                        cam_computer.evaluator.iou_threshold_list,
                    'best_tau_list': cam_computer.evaluator.best_tau_list
                }, fth)

        torch.cuda.empty_cache()

    def plot_perf_curves_top_1_5(self, curves: dict, fdout: str, title: str):

        x_label = r'$\tau$'
        y_label = 'BoxAcc'

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

        for i, top in enumerate(['top1', 'top5']):

            iouthres = sorted(list(curves[top].keys()))
            for iout in iouthres:
                axes[0, i].plot(curves['x'], curves[top][iout],
                                label=r'{}: $\sigma$={}'.format(top, iout))

            axes[0, i].xaxis.set_tick_params(labelsize=5)
            axes[0, i].yaxis.set_tick_params(labelsize=5)
            axes[0, i].set_xlabel(x_label, fontsize=8)
            axes[0, i].set_ylabel(y_label, fontsize=8)
            axes[0, i].grid(True)
            axes[0, i].legend(loc='best')
            axes[0, i].set_title(top)

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                    dpi=300)

    @staticmethod
    def plot_perf_curves(curves: dict, fdout: str, title: str):

        bbox = True
        x_label = r'$\tau$'
        y_label = 'BoxAcc'
        if 'y' in curves:
            bbox = False
            x_label = 'Recall'
            y_label = 'Precision'

        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        if bbox:
            iouthres = sorted([kk for kk in curves.keys() if kk != 'x'])
            for iout in iouthres:
                ax.plot(curves['x'], curves[iout],
                        label=r'$\sigma$={}'.format(iout))
        else:
            ax.plot(curves['x'], curves['y'], color='tab:orange',
                    label='Precision/Recall')

        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True)
        plt.legend(loc='best')
        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_best_epoch(self, split):
        self.args.best_epoch_loc = self.performance_meters[split][
            constants.LOCALIZATION_MTR].best_epoch

        self.args.best_epoch_cl = self.performance_meters[split][
            constants.CLASSIFICATION_MTR].best_epoch

    def save_checkpoints(self, split):
        raise NotImplementedError('do not call it. inconsistent function.')

        best_epoch = self.performance_meters[split][
            constants.LOCALIZATION_MTR].best_epoch

        self._save_model(checkpoint_type=constants.BEST_LOC, epoch=best_epoch)

        best_epoch = self.performance_meters[split][
            constants.CLASSIFICATION_MTR].best_epoch
        self._save_model(checkpoint_type=constants.BEST_CL, epoch=best_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]

        if checkpoint_type == constants.BEST_LOC:
            _model = deepcopy(self.best_model_loc).to(self.cpu_device).eval()
        elif checkpoint_type == constants.BEST_CL:
            _model = deepcopy(self.best_model_cl).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        save_dir = join(self.args.outd_backup, tag)  # save in scratch.
        os.makedirs(save_dir, exist_ok=True)

        if self.args.task == constants.STD_CL:
            to_save = {
                'encoder': _model.encoder.state_dict(),
                'classification_head': _model.classification_head.state_dict()
            }

        elif self.args.task in [constants.F_CL, constants.TCAM]:
            to_save = {
                'encoder': _model.encoder.state_dict(),
                'decoder': _model.decoder.state_dict(),
                'classification_head': _model.classification_head.state_dict(),
                'segmentation_head': _model.segmentation_head.state_dict()
            }

            if _model.reconstruction_head is not None:
                to_save['reconstruction_head'
                ] = _model.reconstruction_head.state_dict()

        elif self.args.task == constants.C_BOX:

            to_save = {
                'encoder': _model.encoder.state_dict(),
                'box_head': _model.box_head.state_dict()
            }

        else:
            raise NotImplementedError

        save_filename = f'{self.current_step}_{constants.CHP_BEST_M}.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(to_save, f=save_path)
        self._save_args(path=join(save_dir, 'config_model.yaml'))
        DLLogger.log(f'Stored best model ({checkpoint_type}) @ {save_path}')

        fn_keep_last_n_checkpoints(
            save_dir, n=self.args.keep_last_n_checkpoints,
            checkpoints_health=self.cpt_best_models_health[checkpoint_type],
            key=constants.CHP_BEST_M
        )

    def model_selection(self, epoch, split):
        assert split == constants.VALIDSET

        if (self.performance_meters[split][constants.LOCALIZATION_MTR]
                .best_epoch) == epoch:
            self.best_model_loc = deepcopy(self._pytorch_model).to(
                self.device).eval()

            self._save_model(checkpoint_type=constants.BEST_LOC,
                             epoch=epoch)

        if (self.performance_meters[split][constants.CLASSIFICATION_MTR]
                .best_epoch) == epoch:
            self.best_model_cl = deepcopy(self._pytorch_model).to(
                self.device).eval()

            self._save_model(checkpoint_type=constants.BEST_CL,
                             epoch=epoch)

        # after successfully storing best model, we store perf. tracker.
        self.save_checkpoint_performance_tracker()

    def load_checkpoint(self, checkpoint_type, net=None):
        if net is None:
            net = self.model

        assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        save_dir = join(self.args.outd_backup, tag)  # load from scratch.
        _, cpt = find_last_checkpoint(save_dir, constants.CHP_BEST_M)

        if self.args.task == constants.STD_CL:
            encoder_w = cpt['encoder']
            classification_head_w = cpt['classification_head']

            if encoder_w is not None:
                encoder_w = move_state_dict_to_device(encoder_w, self.device)
                net.encoder.super_load_state_dict(encoder_w, strict=True)

            if classification_head_w is not None:
                classification_head_w = move_state_dict_to_device(
                    classification_head_w, self.device)
                net.classification_head.load_state_dict(
                    classification_head_w, strict=True)

        elif self.args.task in [constants.F_CL, constants.TCAM]:

            encoder_w = cpt['encoder']
            decoder_w = cpt['decoder']
            classification_head_w = cpt['classification_head']
            segmentation_head_w = cpt['segmentation_head']

            if encoder_w is not None:
                encoder_w = move_state_dict_to_device(encoder_w, self.device)
                net.encoder.super_load_state_dict(encoder_w, strict=True)

            if classification_head_w is not None:
                classification_head_w = move_state_dict_to_device(
                    classification_head_w, self.device)
                net.classification_head.load_state_dict(
                    classification_head_w, strict=True)

            if decoder_w is not None:
                decoder_w = move_state_dict_to_device(decoder_w, self.device)
                net.decoder.load_state_dict(decoder_w, strict=True)

            if segmentation_head_w is not None:
                segmentation_head_w = move_state_dict_to_device(
                    segmentation_head_w, self.device)
                net.segmentation_head.load_state_dict(
                    segmentation_head_w, strict=True)

            if net.reconstruction_head is not None:
                reconstruction_head_w = cpt['reconstruction_head']
                if reconstruction_head_w is not None:
                    reconstruction_head_w = move_state_dict_to_device(
                        reconstruction_head_w, self.device)
                    net.reconstruction_head.load_state_dict(
                        reconstruction_head_w, strict=True)

        elif self.args.task == constants.C_BOX:
            encoder_w = cpt['encoder']
            box_head_w = cpt['box_head']

            if encoder_w is not None:
                encoder_w = move_state_dict_to_device(encoder_w, self.device)
                net.encoder.super_load_state_dict(encoder_w, strict=True)

            if box_head_w is not None:
                box_head_w = move_state_dict_to_device(box_head_w, self.device)
                net.box_head.load_state_dict(box_head_w, strict=True)

        else:
            raise NotImplementedError

    def report_train(self, train_performance, epoch, split=constants.TRAINSET):
        DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
            epoch, split, train_performance['classification_acc']))
        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for metric in self._EVAL_METRICS:
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: {} ".format(
                epoch, split, metric,
                self.performance_meters[split][metric].current_value))
            DLLogger.log("REPORT EPOCH/{}: split: {}/metric {}: "
                         "{}_best ".format(
                          epoch, split, metric,
                          self.performance_meters[split][metric].best_value))

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_meter(self, metrics: dict, filename: str, title: str = '',
                   xlabel: str = '', best_iter: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                if best_iter is not None:
                    axes[i, j].plot([x[best_iter]], [val[best_iter]],
                                    marker='o',
                                    markersize=5,
                                    color="red")
                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    @staticmethod
    def clean_metrics(metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        for split in [constants.TRAINSET, constants.VALIDSET]:
            best_epoch = self.performance_meters[split][
                self._BEST_CRITERION_METRIC].best_epoch
            title = 'DS: {}, Split: {}, box_v2_metric: {}. Best iter.:' \
                    '{} {}'.format(
                     self.args.dataset, split, self.args.box_v2_metric,
                     best_epoch, xlabel)
            filename = '{}-{}-boxv2-{}'.format(
                self.args.dataset, split, self.args.box_v2_metric)
            self.plot_meter(
                self.clean_metrics(meters[split]), filename=filename,
                title=title, xlabel=xlabel, best_iter=best_epoch)


def ddp_barrier(distributed: bool):
    if distributed:
        dist.barrier()