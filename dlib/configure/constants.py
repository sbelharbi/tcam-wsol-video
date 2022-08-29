# possible tasks
STD_CL = "STD_CL"  # standard classification using only the encoder features.
# ouputs: logits, cams.
F_CL = 'F_CL'  # standard classification but using the decoder features.
# outputs: logits, cams.
TCAM = 'TCAM'  # temporal CAM for wsol for videos.
C_BOX = 'C_BOX'  # c-box task: weakly supervised object localization.
SEG = "SEGMENTATION"  # standard supervised segmentation. outputs:
# segmentation masks.

TASKS = [STD_CL, F_CL, C_BOX, TCAM, SEG]


# name of the classifier head (pooling operation)
WILDCATHEAD = "WildCatCLHead"

GAP = 'GAP'
WGAP = 'WGAP'
MAXPOOL = 'MaxPool'
LSEPOOL = 'LogSumExpPool'

SPATIAL_POOLINGS = [WILDCATHEAD, GAP, WGAP, MAXPOOL, LSEPOOL]

# methods
METHOD_WILDCAT = 'WILDCAT'  # pooling: WILDCATHEAD
METHOD_GAP = 'GAP'  # pooling: GAP

METHOD_MAXPOOL = 'MaxPOL'  # pooling: MAXPOOL
METHOD_LSE = 'LogSumExp'  # pooling: logsumexp.

# -- all methods below use WGAP.

METHOD_CAM = 'CAM'
METHOD_SCORECAM = 'ScoreCAM'
METHOD_SSCAM = 'SSCAM'
METHOD_ISCAM = 'ISCAM'

METHOD_GRADCAM = 'GradCam'
METHOD_GRADCAMPP = 'GradCAMpp'
METHOD_SMOOTHGRADCAMPP = 'SmoothGradCAMpp'
METHOD_XGRADCAM = 'XGradCAM'
METHOD_LAYERCAM = 'LayerCAM'


METHODS = [METHOD_WILDCAT,
           METHOD_GAP,
           METHOD_MAXPOOL,
           METHOD_LSE,
           METHOD_CAM,
           METHOD_SCORECAM,
           METHOD_SSCAM,
           METHOD_ISCAM,
           METHOD_GRADCAM,
           METHOD_GRADCAMPP,
           METHOD_SMOOTHGRADCAMPP,
           METHOD_XGRADCAM,
           METHOD_LAYERCAM]

METHOD_2_POOLINGHEAD = {
        METHOD_WILDCAT: WILDCATHEAD,
        METHOD_GAP: GAP,
        METHOD_MAXPOOL: MAXPOOL,
        METHOD_LSE: LSEPOOL,
        METHOD_CAM: WGAP,
        METHOD_SCORECAM: WGAP,
        METHOD_SSCAM: WGAP,
        METHOD_ISCAM: WGAP,
        METHOD_GRADCAM: WGAP,
        METHOD_GRADCAMPP: WGAP,
        METHOD_SMOOTHGRADCAMPP: WGAP,
        METHOD_XGRADCAM: WGAP,
        METHOD_LAYERCAM: WGAP
    }

METHOD_REQU_GRAD = {
        METHOD_WILDCAT: False,
        METHOD_GAP: False,
        METHOD_MAXPOOL: False,
        METHOD_LSE: False,
        METHOD_CAM: False,
        METHOD_SCORECAM: False,
        METHOD_SSCAM: False,
        METHOD_ISCAM: False,
        METHOD_GRADCAM: True,
        METHOD_GRADCAMPP: True,
        METHOD_SMOOTHGRADCAMPP: True,
        METHOD_XGRADCAM: True,
        METHOD_LAYERCAM: True
}

METHOD_LITERAL_NAMES = {
        METHOD_WILDCAT: 'WILDCAT',
        METHOD_GAP: 'GAP',
        METHOD_MAXPOOL: 'MaxPool',
        METHOD_LSE: 'LSEPool',
        METHOD_CAM: 'CAM*',
        METHOD_SCORECAM: 'ScoreCAM',
        METHOD_SSCAM: 'SSCAM',
        METHOD_ISCAM: 'ISCAM',
        METHOD_GRADCAM: 'GradCAM',
        METHOD_GRADCAMPP: 'GradCam++',
        METHOD_SMOOTHGRADCAMPP: 'Smooth-GradCAM++',
        METHOD_XGRADCAM: 'XGradCAM',
        METHOD_LAYERCAM: 'LayerCAM'
}
# datasets mode
DS_TRAIN = "TRAIN"
DS_EVAL = "EVAL"

dataset_modes = [DS_TRAIN, DS_EVAL]

# Tags for samples
L = 0  # Labeled samples

samples_tags = [L]  # list of possible sample tags.

# pixel-wise supervision:
ORACLE = "ORACLE"  # provided by an oracle.
SELF_LEARNED = "SELF-LEARNED"  # self-learned.
VOID = "VOID"  # None

# segmentation modes.
#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


# pretraining
IMAGENET = "imagenet"

# archs
STDCLASSIFIER = "STDClassifier"

UNETFCAM = 'UnetFCAM'  # USED
# UNETCBOX = 'UnetCBox'  #
DENSEBOXNET = 'DenseBoxNet'  # used

UNETTCAM = 'UnetTCAM'  # used

UNET = "Unet"
UNETPLUPLUS = "UnetPlusPlus"
MANET = "MAnet"
LINKNET = "Linknet"
FPN = "FPN"
PSPNET = "PSPNet"
DEEPLABV3 = "DeepLabV3"
DEEPLABV3PLUS = "DeepLabV3Plus"
PAN = "PAN"

ARCHS = [STDCLASSIFIER, UNETFCAM, UNETTCAM, DENSEBOXNET]

# ecnoders

#  resnet
RESNET50 = 'resnet50'

# vgg
VGG16 = 'vgg16'

# inceptionv3
INCEPTIONV3 = 'inceptionv3'

BACKBONES = [RESNET50,
             VGG16,
             INCEPTIONV3
             ]

# ------------------------------------------------------------------------------

# datasets
DEBUG = False


ILSVRC = "ILSVRC"
CUB = "CUB"
OpenImages = 'OpenImages'
# wsol in videos.
YTOV1 = "YouTube-Objects-v1.0"
YTOV22 = "YouTube-Objects-v2.2"
YTBB = "YouTube-BoundingBoxes"

FORMAT_DEBUG = 'DEBUG_{}'
if DEBUG:
    CUB = FORMAT_DEBUG.format(CUB)
    ILSVRC = FORMAT_DEBUG.format(ILSVRC)
    OpenImages = FORMAT_DEBUG.format(OpenImages)
    YTOV1 = FORMAT_DEBUG.format(YTOV1)
    YTOV22 = FORMAT_DEBUG.format(YTOV22)


datasets = [CUB, ILSVRC, OpenImages, YTOV1, YTOV22]

BUCKET_SZ = 8

NBR_CHUNKS_TR = {
    'ILSVRC': 30 * 8,  # 30 *8: ~5k per chunk
    'DEBUG_ILSVRC': 3 * 8,  # 3 *8: ~5k per chunk.
    # no chunking:
    'CUB': -1,
    'DEBUG_CUB': -1,
    'OpenImages': -1,
    'DEBUG_OpenImages': -1,
    'YouTube-Objects-v1.0': -1,
    'DEBUG_YouTube-Objects-v1.0': -1,
    'YouTube-Objects-v2.2': -1,
    'DEBUG_YouTube-Objects-v2.2': -1
}

RELATIVE_META_ROOT = './folds/wsol-done-right-splits'

NUMBER_CLASSES = {
    ILSVRC: 1000,
    CUB: 200,
    OpenImages: 100,
    YTOV1: 10,
    YTOV22: 10
}

CROP_SIZE = 224
RESIZE_SIZE = 256

DS_SHOTS = 'SHOTS'
DS_FRAMES = 'FRAMES'

DS_MODES = [DS_SHOTS, DS_FRAMES]

# ================= check points
BEST_CL = 'best_classification'
BEST_LOC = 'best_localization'

# ==============================================================================

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

# =================================================
NCOLS = 80  # tqdm ncols.

# stages:
STGS_TR = "TRAIN"
STGS_EV = "EVAL"


# datasets:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'
TESTSET_VIDEO_DEMO = 'test-video-demo'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# image range: [0, 1] --> Sigmoid. [-1, 1]: TANH
RANGE_TANH = "tanh"
RANGE_SIGMOID = 'sigmoid'

# ==============================================================================
# cams extractor
TRG_LAYERS = {
            RESNET50: 'encoder.layer4.2.relu3',
            VGG16: 'encoder.relu',
            INCEPTIONV3: 'encoder.SPG_A3_2b.2'
        }
FC_LAYERS = {
    RESNET50: 'classification_head.fc',
    VGG16: 'classification_head.fc',
    INCEPTIONV3: 'classification_head.fc'
}

# EXPs
OVERRUN = False

# cam_curve_interval: for bbox. use high interval for validation (not test).
# high number of threshold slows down the validation because of
# `cv2.findContours`. this gets when cams are bad leading to >1k contours per
# threshold. default evaluation: .001.
VALID_FAST_CAM_CURVE_INTERVAL = .004

# data: name of the folder where cams will be stored.
DATA_CAMS = 'data_cams'

FULL_BEST_EXPS = 'full_best_exps'
PERTURBATIONS_FD = 'perturbations_analysis'

# DDP
NCCL = 'nccl'
GLOO = 'gloo'
MPI = 'mpi'

# CC: communitation folder
SCRATCH_COMM = 'wsol-video/communication'
SCRATCH_FOLDER = 'wsol-video'

# metrics names
LOCALIZATION_MTR = 'localization'
CLASSIFICATION_MTR = 'classification'
FAILD_BOXES_MTR = 'failed boxes'


# norms
NORM1 = '1'
NORM2 = '2'
NORM0EXP = '0EXP'

NORMS = [NORM1, NORM2, NORM0EXP]

# size estimation.
SIZE_DATA = 'size_data'
SIZE_CONST = 'size_constant'

# virtual env name: local.
_ENV_NAME = 'wsol-video'

# time dependency: which frames to consider:
# 1. before current frame (+ instant). 2. after current frame (+instant).
# 3. both (+instant). 4. instant only.

TIME_BEFORE = 'before'
TIME_AFTER = 'after'
TIME_BEFORE_AFTER = 'before-after'
TIME_INSTANT = 'instant'
TIME_DEPENDENCY = [TIME_BEFORE, TIME_AFTER, TIME_BEFORE_AFTER, TIME_INSTANT]

# frames weights
FRAME_WEIGHT_UNIFORM = 'uniform'
FRAME_WEIGHT_EXP = 'exponential'
FRAME_WEIGHT = [FRAME_WEIGHT_EXP, FRAME_WEIGHT_UNIFORM]

# aggregate: cam or bbox.
AGGREGATE_CAMS = 'Aggregate_CAMs'
AGGREGATE_BBOX = 'Aggregate_BBOXs'
AGGREGATION = [AGGREGATE_BBOX, AGGREGATE_CAMS]

# seeding distribution:
SEED_UNIFORM = 'seed_uniform'
SEED_WEIGHTED = 'seed_weighted'
SEED_TECHS = [SEED_UNIFORM, SEED_WEIGHTED]


# checkpoint type:
CHP_M = 'model'
CHP_O = 'optimizer'
CHP_LR = 'lr_scheduler'
CHP_T = 't'  # of elb.
CHP_CP = 'checkpoint'
CHP_TR = 'tracker'
CHP_BEST_M = 'best_model'

CHPS = [CHP_O, CHP_M, CHP_LR, CHP_T, CHP_CP, CHP_TR, CHP_BEST_M]

ROI_ALL = 'roi_all'
ROI_H_DENSITY = 'roi_high_density'
ROI_LARGEST = 'largest'

ROI_SELECT = [ROI_ALL, ROI_H_DENSITY, ROI_LARGEST]

# keys
KEY_CAM_FG_SZ = 'cam_fg_size'
