### Pytorch 1.11.0 code for:
`TCAM: Temporal Class Activation Maps for Object Localization in
Weakly-Labeled Unconstrained Videos`([https://arxiv.
org/abs/2201.02445](https://arxiv.org/abs/2201.02445))

### Citation:
```
@article{tcamsbelharbi2022,
  title={TCAM: Temporal Class Activation Maps for Object Localization in
Weakly-Labeled Unconstrained Videos},
  author={Belharbi, S. and Ben Ayed, I. and McCaffrey, L. and Granger, E.},
  journal={CoRR},
  volume={abs/1909.03354},
}
```

### Issues:
Please create a github issue.

### Content:
* [Method](#method)
* [Results](#results)
* [Requirements](#re2q)
* [Datasets](#datasets)
* [Run code](#run)
* [Localization performance](#locperf)
* [Required changes from your side](#changes)


#### <a name='method'> Method</a>:
<img src="doc/method.png" alt="method" width="600">

<img src="doc/cam-tmp.png" alt="camp-tmp" width="600">

#### <a name='results'> Results</a>:

[![Watch the video](https://i.imgur.com/vKb2F1B.png)](https://youtu.be/vt5fpE0bzSY)

<img src="doc/results.png" alt="results" width="600">


#### <a name='reqs'> Requirements</a>:

See full requirements at [./dependencies/requirements.txt](./dependencies/requirements.txt)

* Python 3.7.10
* [Pytorch](https://github.com/pytorch/pytorch)  1.11.0
* [torchvision](https://github.com/pytorch/vision) 0.12.0
* [Full dependencies](dependencies/requirements.txt)
* Build and install CRF:
    * Install [Swig](http://www.swig.org/index.php)
    * CRF
```shell
cdir=$(pwd)
cd dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install
cd $cdir
cd dlib/crf/crfwrapper/colorbilateralfilter
swig -python -c++ colorbilateralfilter.i
python setup.py install
```

#### <a name="datasets"> Download datasets </a>:
See [folds/wsol-done-right-splits/dataset-scripts](
folds/wsol-done-right-splits/dataset-scripts). For more details, see
[wsol-done-right](https://github.com/clovaai/wsolevaluation) repo.

You can use these scripts to download the datasets: [cmds](./cmds). Use the
script [_video_ds_ytov2_2.py](./dlib/datasets/_video_ds_ytov2_2.py) to
reformat YTOv2.2.

Once you download the datasets, you need to adjust the paths in
[get_root_wsol_dataset()](dlib/configure/config.py).

#### <a name="datasets"> Run code </a>:
Download files in `download-files.txt` from google drive.

1. WSOL baselines: CAM over YouTube-Objects-v1.0 using ResNet50:
```shell
cudaid=0  # cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

getfreeport() {
freeport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}
export OMP_NUM_THREADS=50
export NCCL_BLOCKING_WAIT=1
plaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
getfreeport
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port=$freeport main.py --local_world_size=1 \
       --task STD_CL \
       --encoder_name resnet50 \
       --arch STDClassifier \
       --opt__name_optimizer sgd \
       --dist_backend gloo \
       --batch_size 32 \
       --max_epochs 100 \
       --checkpoint_save 100 \
       --keep_last_n_checkpoints 10 \
       --freeze_cl False \
       --freeze_encoder False \
       --support_background True \
       --method CAM \
       --spatial_pooling WGAP \
       --dataset YouTube-Objects-v1.0 \
       --box_v2_metric False \
       --cudaid $cudaid \
       --amp True \
       --plot_tr_cam_progress False \
       --opt__lr 0.001 \
       --opt__step_size 15 \
       --opt__gamma 0.9 \
       --opt__weight_decay 0.0001 \
       --exp_id 08_28_2022_11_51_57_590148__5889160
```
Train until convergence, then store the cams of trainset to be used later.
From the experiment folder, copy both folders 'YouTube-Objects-v1.0-resnet50-CAM-WGAP-cp_best_localization-boxv2_False'
and 'YouTube-Objects-v1.0-resnet50-CAM-WGAP-cp_best_classification
-boxv2_False' to the folder 'pretrained'. The contain best weights which
will be loaded by TCAM model.

2. TCAM: Run:
```shell
cudaid=0  # cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

getfreeport() {
freeport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
}
export OMP_NUM_THREADS=50
export NCCL_BLOCKING_WAIT=1
plaunch=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
getfreeport
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port=$freeport main.py --local_world_size=1 \
       --task TCAM \
       --encoder_name resnet50 \
       --arch UnetTCAM \
       --opt__name_optimizer sgd \
       --dist_backend gloo \
       --batch_size 32 \
       --max_epochs 100 \
       --checkpoint_save 100 \
       --keep_last_n_checkpoints 10 \
       --freeze_cl True \
       --support_background True \
       --method CAM \
       --spatial_pooling WGAP \
       --dataset YouTube-Objects-v1.0 \
       --box_v2_metric False \
       --cudaid $cudaid \
       --amp True \
       --plot_tr_cam_progress False \
       --opt__lr 0.01 \
       --opt__step_size 15 \
       --opt__gamma 0.9 \
       --opt__weight_decay 0.0001 \
       --elb_init_t 1.0 \
       --elb_max_t 10.0 \
       --elb_mulcoef 1.01 \
       --sl_tc True \
       --sl_tc_knn 1 \
       --sl_tc_knn_mode before \
       --sl_tc_knn_t 0.0 \
       --sl_tc_knn_epoch_switch_uniform -1 \
       --sl_tc_min_t 0.0 \
       --sl_tc_lambda 1.0 \
       --sl_tc_min 1 \
       --sl_tc_max 1 \
       --sl_tc_ksz 3 \
       --sl_tc_max_p 0.6 \
       --sl_tc_min_p 0.1 \
       --sl_tc_seed_tech seed_weighted \
       --sl_tc_use_roi True \
       --sl_tc_roi_method roi_all \
       --sl_tc_roi_min_size 0.05 \
       --crf_tc True \
       --crf_tc_lambda 2e-09 \
       --crf_tc_sigma_rgb 15.0 \
       --crf_tc_sigma_xy 100.0 \
       --crf_tc_scale 1.0 \
       --max_sizepos_tc True \
       --max_sizepos_tc_lambda 0.01 \
       --size_bg_g_fg_tc False \
       --empty_out_bb_tc False \
       --sizefg_tmp_tc False \
       --knn_tc 0 \
       --rgb_jcrf_tc False \
       --exp_id 08_28_2022_11_50_04_936875__7685436
```
