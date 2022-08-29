import os
import sys
from os.path import join, dirname, abspath
import datetime as dt


import munch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.tools import chunk_it
import dlib.dllogger as DLLogger

__all__ = ['get_config']


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_data_paths(args, dsname=None):
    if dsname is None:
        dsname = args['dataset']

    train = val = test = join(args['data_root'], dsname)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = "{}/wsol-done-right".format(os.environ["DATASETSH"])
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = "{}/datasets".format(os.environ["DATASETSH"])
        # todo: tmp. clean later.
        elif os.environ['HOST_XXX'] == 'lab_josi':
            baseurl = "{}/Datasets".format(os.environ["DATASETSH"])

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            # todo: tmp. clean later.
            if "SLURM_TMPDIR_JOS" in os.environ.keys():
                baseurl = "{}/Datasets".format(
                    os.environ["SLURM_TMPDIR_JOS"])
            else:
                baseurl = "{}/datasets/wsol-done-right".format(
                    os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this case my happen if someone calls this function outside a job.
            baseurl = "{}/datasets/wsol-done-right".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are enable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_nbr_bucket(ds: str) -> int:
    nbr_chunks = constants.NBR_CHUNKS_TR[ds]
    out = chunk_it(list(range(nbr_chunks)), constants.BUCKET_SZ)
    return len(list(out))


def get_config(ds):
    assert ds in constants.datasets, ds

    args = {
        # ======================================================================
        #                               GENERAL
        # ======================================================================
        "MYSEED": 0,  # Seed for reproducibility. int >= 0.
        "cudaid": '0',  # str. cudaid. form: '0,1,2,3' for cuda devices.
        'num_gpus': 1,  # number of gpus. will be ste automatically.
        "debug_subfolder": '',  # subfolder used for debug. if '', we do not
        # consider it.
        "dataset": ds,  # name of the dataset.
        "num_classes": constants.NUMBER_CLASSES[ds],  # Total number of classes.
        "crop_size": constants.CROP_SIZE,  # int. size of cropped patch.
        "resize_size": constants.RESIZE_SIZE,  # int. size to which the image
        # is resized before cropping.
        "batch_size": 8,  # the batch size for training.
        "batch_size_backup": 8,  # backup for 'batch_size'. (copy of original
        # value)
        "num_workers": 4,  # number of workers for dataloader of the trainset.
        "exp_id": "123456789",  # exp id. random number unique for the exp.
        "verbose": True,  # if true, we log messages right away. otherwsie,
        # we flush them at the end.
        'plot_tr_cam_progress': False,  # if true, cam progress of selected
        # samples will be plotted. this will add more time. watch out.
        'plot_tr_cam_progress_n': 0,  # int.  how many samples to consider
        # for plotting.
        'fd_exp': None,  # relative path to folder where the exp.
        'abs_fd_exp': None,  # absolute path to folder where the exp.
        'best_epoch_loc': 0,  # int. best epoch for localization task.
        'best_epoch_cl': 0,  # int. best epoch for classification task.
        'img_range': constants.RANGE_TANH,  # range of the image values after
        # normalization either in [0, 1] or [-1, 1]. see constants.
        't0': dt.datetime.now(),  # approximate time of starting the code.
        'tend': None,  # time when this code ends.
        'running_time': None,  # the time needed to run the entire code.
        'ds_chunkable': (constants.NBR_CHUNKS_TR[ds] != -1),  # whether the
        # trainset is chunked or not. only ilsvrc is chunked. if you want to
        # turn off this completely, set it to False.
        'nbr_buckets': get_nbr_bucket(ds),  # number of train bucket. applied
        # only for chunkable datasets.
        # ======================================================================
        #                      WSOL DONE RIGHT
        # ======================================================================
        "data_root": get_root_wsol_dataset(),  # absolute path to data parent
        # folder.
        "metadata_root": constants.RELATIVE_META_ROOT,  # path to metadata.
        # contains splits.
        "mask_root": get_root_wsol_dataset(),  # path to masks.
        "proxy_training_set": False,  # efficient hyper-parameter search with
        # a proxy training set. true/false.
        "std_cams_folder": mch(train='', val='', test=''),  # folders where
        # cams of std_cl are stored to be used for f_cl/c_box training.
        # typicaly, we store only training. this is an option since f_cl/c_box
        # can still compute the std_cals. but, storing them making their
        # access fast  to avoid re-computing them every time during training.
        # the exact location will be determined during parsing the input.
        # this is optional. if we do not find this folder, we recompute the
        # cams. This version: for c-box, we have to use stored cams so we
        # dont use a classifier.
        "std_cams_thresh_file": mch(train='', val='', test=''),  # where ROI
        # thresholds where stored. used for TCAM. this will be set
        # automatically. if not found in right place, we will estimate
        # thresholds automatically same as 'std_cams_folder'.
        "num_val_sample_per_class": 0,  # number of full_supervision
        # validation sample per class. 0 means: use all available samples.
        'cam_curve_interval': .001,  # CAM curve interval.
        'multi_contour_eval': True,  # Bounding boxes are extracted from all
        # contours in the thresholded score map. You can use this feature by
        # setting multi_contour_eval to True (default). Otherwise,
        # bounding boxes are extracted from the largest connected
        # component of the score map.
        'multi_iou_eval': True,
        'iou_threshold_list': [30, 50, 70],
        'box_v2_metric': False,
        'eval_checkpoint_type': constants.BEST_LOC,  # just for
        # stand-alone inference. during training+inference, we evaluate both.
        # Necessary s well for the task F_CL during training to select the
        # init-model-weights-classifier.
        # ======================================================================
        #                      VISUALISATION OF REGIONS OF INTEREST
        # ======================================================================
        "alpha_visu": 100,  # transparency alpha for cams visualization. low is
        # opaque (matplotlib).
        "height_tag": 60,  # the height of the margin where the tag is written.
        # ======================================================================
        #                             OPTIMIZER (n0)
        #                            TRAIN THE MODEL
        # ======================================================================
        "checkpoint_save": 5000,  # frequency of checkpointing [iterations].
        'save_dir_models': 'checkpoints',  # folder's name where to store
        # models (checkpoints).
        'keep_last_n_checkpoints': 2,  # maximum last checkpoints to store.
        "synch_scratch_epoch_freq": 50,  # frequency of synchronizing
        # scratch folder. applied only for CC server. [epochs]
        "slurm_dir": 'slurm',  # folder where to store slurm files.
        "slurm_path1": '',  # absolute path to slurm config file: main file.
        "slurm_path2": '',  # absolute path to slurm config file: command file.
        "optimizer": {  # the optimizer
            # ==================== SGD =======================
            "opt__name_optimizer": "sgd",  # str name. 'sgd', 'adam'
            "opt__lr": 0.001,  # Initial learning rate.
            "opt__momentum": 0.9,  # Momentum.
            "opt__dampening": 0.,  # dampening.
            "opt__weight_decay": 1e-4,  # The weight decay (L2) over the
            # parameters.
            "opt__nesterov": True,  # If True, Nesterov algorithm is used.
            # ==================== ADAM =========================
            "opt__beta1": 0.9,  # beta1.
            "opt__beta2": 0.999,  # beta2
            "opt__eps_adam": 1e-08,  # eps. for numerical stability.
            "opt__amsgrad": False,  # Use amsgrad variant or not.
            # ========== LR scheduler: how to adjust the learning rate. ========
            "opt__lr_scheduler": True,  # if true, we use a learning rate
            # scheduler.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "opt__name_lr_scheduler": "mystep",  # str name.
            "opt__step_size": 40,  # Frequency of which to adjust the lr.
            "opt__gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "opt__last_epoch": -1,  # the index of the last epoch where to stop
            # adjusting the LR.
            "opt__min_lr": 1e-7,  # minimum allowed value for lr.
            "opt__t_max": 100,  # T_max for cosine schedule.
            "opt__lr_classifier_ratio": 10.,  # Multiplicative factor on the
            # classifier layer (head) learning rate.
        },
        # ======================================================================
        #                              MODEL
        # ======================================================================
        "model": {
            "arch": constants.DENSEBOXNET,  # name of the model.
            # see: constants.nets.
            "encoder_name": constants.RESNET50,  # backbone for task of SEG.
            "encoder_weights": constants.IMAGENET,
            # pretrained weights or 'None'.
            "in_channels": 3,  # number of input channel.
            "path_pre_trained": None,
            # None, `None` or a valid str-path. if str,
            # it is the absolute/relative path to the pretrained model. This can
            # be useful to resume training or to force using a filepath to some
            # pretrained weights.
            "strict": True,  # bool. Must be always be True. if True,
            # the pretrained model has to have the exact architecture as this
            # current model. if not, an error will be raise. if False, we do the
            # best. no error will be raised in case of mismatch.
            "support_background": True,  # useful for classification tasks only:
            # std_cl, f_cl only. if true, an additional cam is used for the
            # background. this does not change the number of global
            # classification logits. irrelevant for segmentation task.
            "scale_in": 1.,  # float > 0.  how much to scale
            # the input image to not overflow the memory. This scaling is done
            # inside the model on the same device as the model.
            "freeze_cl": False,  # applied ONLY for task F_CL/TCAM. if true,
            # the classifier (encoder + head) is frozen. the classifier
            # means something else in the task C_BOX, because the classifier
            # is a different model. for C_BOX, the classifier is
            # automatically frozen. IF YOU WANT TO FREEZE THE ENCODER OF THE
            # C_BOX, USE: 'freeze_encoder'. for c_box, we USE an encoder
            # that comes from a pretrained classifier (folder_pre_trained_cl).
            'freeze_encoder': False,  # ONLY FOR C_BOX TASK. if true,
            # the encoder of the localizer (not classifier) is frozen. helpful
            # when the encoder is pretrained. it allows low memory usage,
            # and fast training.
            'scale_domain': 1.,  # for C_BOX ONLY. used to scale (better
            # down) the x- and y-axis length for the box head prediction.
            # This determines the output diemnsion of the box head. for
            # example, for height = 224, and scale = .5, the box head will
            # predict classification of (224 * .5) classes. This allows to
            # reduce the dimension of coordinates probabilities.
            "folder_pre_trained_cl": None,
            # NAME of folder containing weights of
            # classifier. it must be in 'pretrained' folder.
            # used in combination with `freeze_cl`. the folder contains
            # encoder.pt, head.pt weights of the encoder and head. the base name
            # of the folder is a tag used to make sure of compatibility between
            # the source (source of weights) and target model (to be frozen).
            # You do not need to set this parameters if `freeze_cl` is true.
            # we set it automatically when parsing the parameters.
            "folder_pre_trained_seeder": None  # same as
            # folder_pre_trained_cl but the model will be used for generating
            # localization seeds. will be set automatically.
        },
        # ======================================================================
        #                    CLASSIFICATION SPATIAL POOLING
        # ======================================================================
        "method": constants.METHOD_WILDCAT,
        "spatial_pooling": constants.WILDCATHEAD,
        # ======================================================================
        #                        SPATIAL POOLING:
        #                            WILDCAT
        # ======================================================================
        "wc_modalities": 5,
        "wc_kmax": 0.5,
        "wc_kmin": 0.1,
        "wc_alpha": 0.6,
        "wc_dropout": 0.0,
        # ================== LSE pooling
        "lse_r": 10.,  # r for logsumexp pooling.
        # ======================================================================
        #                          Segmentation mode
        # ======================================================================
        "seg_mode": constants.BINARY_MODE,
        # SEGMENTATION mode: bin only always.
        "task": constants.STD_CL,  # task: standard classification,
        # full classification (FCAM).
        "multi_label_flag": False,
        # whether the dataset has multi-label or not.
        # ======================================================================
        #                          ELB
        # ==========================================================================
        "elb_init_t": 1.,  # used for ELB.
        "elb_max_t": 10.,  # used for ELB.
        "elb_mulcoef": 1.01,  # used for ELB.
        # ======================================================================
        #                            CONSTRAINTS:
        #                     'SuperResolution', sr
        #                     'ConRanFieldFcams', crf_fc
        #                     'EntropyFcams', entropy_fc
        #                     'PartUncerknowEntropyLowCams', partuncertentro_lc
        #                     'PartCertKnowLowCams', partcert_lc
        #                     'MinSizeNegativeLowCams', min_sizeneg_lc
        #                     'MaxSizePositiveLowCams', max_sizepos_lc
        #                     'MaxSizePositiveFcams' max_sizepos_fc
        # ======================================================================
        "max_epochs": 150,  # number of training epochs.
        # -----------------------  FCAM
        "sl_fc": False,  # use self-learning over fcams.
        "sl_fc_lambda": 1.,  # lambda for self-learning over fcams
        "sl_start_ep": 0,  # epoch when to start sl loss.
        "sl_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_min": 10,  # int. number of pixels to be considered
        # background (after sorting all pixels).
        "sl_max": 10,  # number of pixels to be considered
        # foreground (after sorting all pixels).
        "sl_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_ksz": 1,  # int, kernel size for dilation around the pixel. must be
        # odd number.
        'sl_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_fg_erode_k': 11,  # int. size of erosion kernel to clean foreground.
        'sl_fg_erode_iter': 1,  # int. number of erosions for foreground.
        # ----------------------- FCAM
        "crf_fc": False,  # use or not crf over fcams.  (penalty)
        "crf_lambda": 2.e-9,  # crf lambda
        "crf_sigma_rgb": 15.,
        "crf_sigma_xy": 100.,
        "crf_scale": 1.,  # scale factor for input, segm.
        "crf_start_ep": 0,  # epoch when to start crf loss.
        "crf_end_ep": -1,  # epoch when to stop using crf loss. -1: never stop.
        # ======================================================================
        # ======================================================================
        #                                EXTRA
        # ======================================================================
        # ======================================================================
        # ----------------------- FCAM
        "entropy_fc": False,  # use or not the entropy over fcams. (penalty)
        "entropy_fc_lambda": 1.,
        # -----------------------  FCAM
        "max_sizepos_fc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_fc_lambda": 1.,
        "max_sizepos_fc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_fc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        # ----------------------------------------------------------------------
        # ----------------------- NOT USED
        # ------------------------------- GENERIC
        "im_rec": False,  # image reconstruction loss.
        "im_rec_lambda": 1.,
        "im_rec_elb": False,  # use or not elb for image reconstruction.
        # ----------------------------- NOT USED
        # ----------------------------------------------------------------------
        # ======================================================================
        # ======================================================================
        # ======================================================================

        # ======================================================================
        #                               TCAM
        # ======================================================================
        "tcam_pretrained_cl_ch_pt": constants.BEST_CL,  # check point for
        # classifier weights. these weights will be loaded as encoder weights
        # of unet. this allows the best classification accuracy.
        "tcam_pretrained_seeder_ch_pt": constants.BEST_LOC,  # check point for
        # model [classifier] used to sample seeds. this model will be used to
        # generate localization seeds.
        # -----------------------  TCAM
        "knn_tc": 0,  # int, number of adjacent frames. left: knn. right: knn.
        # total: knn * 2 + 1 for each frame. e.g. knn_tc=1 means we tke one
        # frame before and one frame after the current frame. knn_tc=0 means
        # learning from single frames. knn_tc > 0, meaning we learn from a
        # set of frames at once. this applies only for trainset. evaluation
        # is done using single frame.
        "sl_tc": False,  # use self-learning over tcams.
        "sl_tc_knn": 0,  # temporal cams. how many cams to consider to
        # estimate the cam to sample from. 0: means look only to the current
        # cam.
        "sl_tc_knn_mode": constants.TIME_INSTANT,  # time dependency for
        # sl_tc_knn. if 'instant', 'sl_tc_knn' must be 0.
        "sl_tc_knn_t": 0.0,  # heating factor. used to overheat cams when
        # using temporal information. val >= 0. if 0, it is not used.
        "sl_tc_knn_epoch_switch_uniform": -1,  # epoch when to to switch
        # sampling to uniform. if -1, it is not considered. if it is
        # different from -1, the value sl_tc_knn_t will be decreased to
        # sl_tc_min_t.
        # linearly; reaching  sl_tc_min_t at epoch
        # sl_tc_knn_epoch_switch_uniform.
        # todo: change to always.
        "sl_tc_min_t": 0.0,  # when decaying t, this is minval.
        "sl_tc_epoch_switch_to_sl": -1,  # epoch when we switch getting seeds
        # from cams of pretrained classifier to the cams of decoder. -1:
        # never do it. when we switch, 'sl_tc_knn' will be instant only.
        "sl_tc_roi_method": constants.ROI_ALL,  # how to get roi from cams.
        # all: take all rois. high density: take only high density.
        "sl_tc_roi_min_size": 5/100.,  # minimal area for roi to be
        # considered. (% in [0, 1])
        "sl_tc_lambda": 1.,  # lambda for self-learning over tcams
        "sl_tc_start_ep": 0,  # epoch when to start sl loss.
        "sl_tc_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_tc_min": 10,  # int. number of pixels to be used as
        # background (after sorting all pixels).
        "sl_tc_max": 10,  # number of pixels to be used as foreground (after
        # sorting all pixels).
        "sl_tc_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_tc_ksz": 1,  # int, kernel size for dilation around the pixel.
        # must be
        # odd number.
        'sl_tc_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_tc_max_p': .2,  # percentage of pixels to be considered for
        # foreground sampling. percentage from entire image size. ROI is
        # determined via thresholding if roi is used.
        'sl_tc_use_roi': False,  # if true, binary roi are estimated. then fg
        # pixels are sampled only from them, guided by cam activations.
        'sl_tc_seed_tech': constants.SEED_UNIFORM,  # how to sample fg.
        # Uniformly or using Bernoulli. for bg: uniform.
        'sl_tc_fg_erode_k': 11,  # int. size of erosion kernel to clean
        # foreground.
        'sl_tc_fg_erode_iter': 0,  # int. number of erosions for foreground.
        # ----------------------- TCAM
        "crf_tc": False,  # use or not crf over tcams.  (penalty)
        "crf_tc_lambda": 2.e-9,  # crf lambda
        "crf_tc_sigma_rgb": 15.,
        "crf_tc_sigma_xy": 100.,
        "crf_tc_scale": 1.,  # scale factor for input, segm.
        "crf_tc_start_ep": 0,  # epoch when to start crf loss.
        "crf_tc_end_ep": -1,  # epoch when to stop using crf loss. -1: never
        # stop.
        "rgb_jcrf_tc": False,  # use or not joint crf over cams over
        # multiple images. apply only color penalty.
        "rgb_jcrf_tc_lambda": 2.e-9,  # crf lambda
        "rgb_jcrf_tc_sigma_rgb": 15.,
        "rgb_jcrf_tc_scale": 1.,  # scale factor for input, segm.
        "rgb_jcrf_tc_start_ep": 0,  # epoch when to start crf loss.
        "rgb_jcrf_tc_end_ep": -1,  # epoch when to stop using crf loss.
        # -1: never stop.
        # ---------- TCAM: max size fg and bg.
        "max_sizepos_tc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_tc_lambda": 1.,
        "max_sizepos_tc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_tc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        "size_bg_g_fg_tc": False,  # size: bg > fg. tcams (elb)
        "size_bg_g_fg_tc_lambda": 1.,  # lambda
        "size_bg_g_fg_tc_start_ep": 0,  # epoch when to start loss.
        "size_bg_g_fg_tc_end_ep": -1,  # epoch when to stop loss. -1:
        # never stop.
        "empty_out_bb_tc": False,  # empty area outside bbox. tcams (elb)
        "empty_out_bb_tc_lambda": 1.,  # lambda
        "empty_out_bb_tc_start_ep": 0,  # epoch when to start loss.
        "empty_out_bb_tc_end_ep": -1,  # epoch when to stop loss. -1:
        # never stop.
        "sizefg_tmp_tc": False,  # estimate size of object using neighbors
        # frames. tcam, elb.
        "sizefg_tmp_tc_knn": 0,  # temporal cams. how many cams to consider to
        # estimate the fg size. 0: means look only to the current
        # cam.
        "sizefg_tmp_tc_knn_mode": constants.TIME_INSTANT,  # time dependency for
        # sizefg_tmp_tc_knn. if 'instant', 'sizefg_tmp_tc_knn' must be 0.
        "sizefg_tmp_tc_eps": 0.001,  # epsilon. small size perturbation to
        # compute bounds.
        "sizefg_tmp_tc_lambda": 1.,
        "sizefg_tmp_tc_start_ep": 0,  # epoch when to start loss.
        "sizefg_tmp_tc_end_ep": -1,  # epoch when to stop using loss. -1:
        # never stop.
        # todo: temporal.
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ GENERIC
        'seg_ignore_idx': -255,  # ignore index for segmentation alignment.
        'amp': False,  # if true, use automatic mixed-precision for training
        'amp_eval': False,  # if true, amp is used for inference.
        # ======================================================================
        #                             DDP:
        # NOT CC(): means single machine.  CC(): multiple nodes.
        # ======================================================================
        'local_rank': 0,  # int. for not CC(). auto-set.
        'local_world_size': 1,  # int. for not CC(). number of gpus to use.
        'rank': 0,  # int. global rank. useful for CC(). 0 otherwise. will be
        # set automatically.
        'init_method': '',  # str. CC(). init method. needs to be defined.
        # will be be determined automatically.
        'dist_backend': constants.GLOO,  # str. CC() or not CC(). distributed
        # backend.
        'world_size': 1,  # init. CC(). total number of gpus. will be
        # determined automatically.
        'is_master': False,  # will be set automatically if this process is
        # the master.
        'is_node_master': False,  # will be set auto. true if this process is
        # has local rank = 0.
        'c_cudaid': 0,  # int. current cuda id. auto-set.
        'distributed': False,  # bool. auto-set. indicates whether we are
        # using ddp or not. This will help differentiate when accessing to
        # model.attributes when it is wrapped with a ddp and when not.
        # ======================================================================
        #                         C-BOX: cb_*
        # ======================================================================
        "cb_pretrained_cl_ch_pt": constants.BEST_CL,
        #
        #
        'cb_area_box': False,  # bool. use/no constraint over size box.
        'cb_area_box_l': 1.,  # lambda.
        'cb_area_normed': False,  # bool. normalized area in [0, 1] or not.
        'cb_area_box_start_epoch': 0,  # start epoch.
        'cb_area_box_end_epoch': -1,  # end epoch
        #
        'cb_cl_score': False,  # bool. use/no constraint score over fg/bg.
        'cb_cl_score_l': 1.,  # lambda.
        'cb_cl_score_start_epoch': 0,  # start epoch.
        'cb_cl_score_end_epoch': -1,  # end epoch
        'cb_cl_score_blur_ksize': 65,  # int. odd. kernel size for blurring.
        'cb_cl_score_blur_sigma': 60.,  # float. Gaussian variance for
        # kernel. use large variance for effective blurring.
        #
        'cb_pp_box': False,  # whether to use or not the previous box
        # constraint. must be true to kick-start the training.
        'cb_pp_box_l': 1.,
        'cb_pp_box_start_epoch': 0,
        'cb_pp_box_end_epoch': -1,
        'cb_pp_box_alpha': .1,  # shrinking factor lower limit.
        'cb_pp_box_min_size_type': constants.SIZE_DATA,  # whether to
        # estimate sizes from valid set or use constant 'cb_pp_box_min_size'.
        'cb_pp_box_min_size': .5,  # constant min size. [0, 1[
        #
        'cb_seed': False,  # use dist. over seeds of foreground.
        'cb_seed_l': 1.,  # lambda.
        'cb_seed_start_epoch': 0,  # start epoch.
        'cb_seed_end_epoch': -1,  # end epoch.
        'cb_seed_erode_k': 11,  # int. size of erosion kernel to clean
        # foreground.
        'cb_seed_erode_iter': 1,  # int. number of erosions for
        # foreground.
        'cb_seed_ksz': 3,  # dilation kernel.
        'cb_seed_n': 1,  # int. number of pixels to sample (fg/bg).
        'cb_seed_bg_low_z': .3,  # lower bound of size to be considered as
        # background.
        'cb_seed_bg_up_z': .4,  # upper bound of size to be considered as
        # background. a size is sampled randomly between [low, up].
        'cb_seed_bg_z_type': constants.SIZE_DATA,  # whether low_z, u_z is
        # estimated from validset or use above constants.
        # ---------------------------------------
        'cb_init_box_size': .95,  # size of the initial box. (mean)
        'cb_init_box_var': .015,  # variance of the initial box size. (var)
    }

    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    args['data_paths'] = configure_data_paths(args, dsname)
    args['metadata_root'] = join(args['metadata_root'], args['dataset'])

    openimg_ds = constants.OpenImages
    if openimg_ds.startswith(pre):
        openimg_ds = dsname.replace('{}_'.format(pre), '')
    args['mask_root'] = join(args['mask_root'], openimg_ds)

    data_cams = join(root_dir, constants.DATA_CAMS)
    if not os.path.isdir(data_cams):
        os.makedirs(data_cams, exist_ok=True)

    return args


if __name__ == '__main__':
    args = get_config(constants.ILSVRC)
    print(args['metadata_root'])
