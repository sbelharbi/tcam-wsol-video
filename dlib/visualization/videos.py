import os
import sys
from os.path import dirname, abspath, join, basename
from typing import Optional, Tuple, List
import fnmatch
import argparse

import cv2
from PIL import Image
from tqdm import tqdm


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.configure.config import get_root_wsol_dataset
from dlib.utils.shared import find_files_pattern


def list_file_names_extension(fd_path: str, pattern_ext: str) -> List[str]:
    out = []
    content = next(os.walk(fd_path))[2]
    for item in content:
        path = join(fd_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, pattern_ext):
            out.append(item)

    out = sorted(out, reverse=False)
    return out


def _build_video_from_frames(lframes: list,
                            shot_folder: str,
                            fps: int,
                            w: int, h: int,
                            output_v_path: str,  # no extension.
                            delete_frames: bool,
                            delete_folder_shot: bool):
    ext = '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(
        f'{output_v_path}{ext}', fourcc, fps, (w, h))

    for frame in lframes:
        video.write(cv2.resize(cv2.imread(frame), (w, h),
                               interpolation=cv2.INTER_AREA))

        if delete_frames:
            os.system(f'rm {frame}')

    cv2.destroyAllWindows()
    video.release()


def build_all_videos_ytov1_0_demo_test(path: str, fps: int, ds: str, split:str):
    assert ds in [constants.YTOV1, constants.YTOV22], ds
    assert split in [constants.TRAINSET, constants.TESTSET,
                     constants.TESTSET_VIDEO_DEMO], split


    folds = join(root_dir, constants.RELATIVE_META_ROOT, ds, split)

    with open(join(folds, 'image_ids.txt'), 'r') as fin:
        content = fin.readlines()
        content = [s.strip('\n') for s in content]

        _content = [join(path, f'{l}.png') for l in content]

        if os.path.isfile(_content[0]):  # frame mode.
            content = _content
        else:  # shot mode.
            _content = [join(path, l) for l in content]
            content = []
            for shot in _content:
                assert os.path.isdir(shot), shot
                files = find_files_pattern(shot, '*.png')
                content.extend(files)


    shots = dict()
    for l in content:
        shot = dirname(l)
        # img = Image.open(l, 'r').convert('RGB')
        # w, h = img.size
        w, h = 500, 250

        if shot in shots:
            shots[shot]['frames'].append((l, basename(l)))
            shots[shot]['sizes'].append([w, h])
        else:
            shots[shot] = {
                'frames': [(l, basename(l))],
                'sizes': [[w, h]]
            }

    n = len(list(shots.keys()))

    for shot in tqdm(shots, ncols=80, total=n):

        video_path = shot.rstrip(os.sep)
        for sep in ['/', '\\']:
            video_path = video_path.rstrip(
                sep)  # depends where
            # folds where generated.

        lframes = shots[shot]['frames']
        # sort frames: 0, 1, ...
        lframes = sorted(lframes, key=lambda tup: tup[1],
                         reverse=False)
        _lframes = [v[0] for v in lframes]

        _build_video_from_frames(lframes=_lframes,
                                 output_v_path=video_path,
                                 shot_folder=shot, fps=fps,
                                 w=w, h=h, delete_frames=False,
                                 delete_folder_shot=False)


def rename(fd: str):
    lfiles = list_file_names_extension(fd_path=fd, pattern_ext='*.png')
    # lfiles = find_files_pattern(fd, '*.png')
    # lfiles = [basename(f) for f in lfiles]

    for f in lfiles:
        pnew = join(fd, f.replace('_', os.sep))
        if not pnew.endswith('.jpg.png'):
            pnew = pnew.replace('.png', '.jpg.png')

        d = dirname(pnew)
        os.makedirs(d, exist_ok=True)
        if join(fd, f) != pnew:
            os.rename(join(fd, f), pnew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str,
                        default=constants.TESTSET_VIDEO_DEMO,
                        help="Split: train, test_demo.")
    parser.add_argument("--dataset", type=str,
                        default=constants.YTOV1,
                        help="Dataset.")
    _PATHS = {
        constants.YTOV1:
            {
                constants.TRAINSET:"full_best_exps/YouTube-Objects-v1.0/resnet50"
                                "/TCAM/LayerCAM/id_07_01_2022_14_09_53_612298__7416821-tsk_TCAM-ds_YouTube-Objects-v1.0-mth_LayerCAM-spooling_WGAP-sd_0-ecd_resnet50-box_v2_metric_False-sl_tc_yes-seed_tech_seed_weighted-crf_tc_yes-max_sizepos_tc_yes/best_localization/train/vizu/50",
                constants.TESTSET_VIDEO_DEMO: "full_best_exps/YouTube-Objects-v1.0/resnet50"
                                "/TCAM/LayerCAM/id_07_01_2022_14_09_53_612298__7416821-tsk_TCAM-ds_YouTube-Objects-v1.0-mth_LayerCAM-spooling_WGAP-sd_0-ecd_resnet50-box_v2_metric_False-sl_tc_yes-seed_tech_seed_weighted-crf_tc_yes-max_sizepos_tc_yes/best_localization/test-video-demo/vizu/50"
            },
        constants.YTOV22:
            {
                constants.TRAINSET:
                    "full_best_exps/YouTube-Objects-v2.2/resnet50"
                                    "/TCAM/LayerCAM/id_07_31_2022_11_51_38_064351__797444-tsk_TCAM-ds_YouTube-Objects-v2.2-mth_LayerCAM-spooling_WGAP-sd_0-ecd_resnet50-box_v2_metric_False-sl_tc_yes-seed_tech_seed_weighted-crf_tc_yes-max_sizepos_tc_yes/best_localization/train/vizu/50",
                constants.TESTSET_VIDEO_DEMO:
                    "full_best_exps/YouTube-Objects-v2.2/resnet50"
                                              "/TCAM/LayerCAM/id_07_31_2022_11_51_38_064351__797444-tsk_TCAM-ds_YouTube-Objects-v2.2-mth_LayerCAM-spooling_WGAP-sd_0-ecd_resnet50-box_v2_metric_False-sl_tc_yes-seed_tech_seed_weighted-crf_tc_yes-max_sizepos_tc_yes/best_localization/test-video-demo/vizu/50"
            }
    }
    parsed_args = parser.parse_args()
    split = parsed_args.split
    dataset = parsed_args.dataset

    r_path = _PATHS[dataset][split]

    path = join(root_dir, r_path)
    # if not os.path.isdir(path):
    #     os.system(f'tar -xvf {path}.tar.gz')
    # rename(path)
    # Youtube Objects videos V1.0

    build_all_videos_ytov1_0_demo_test(path=path, fps=15, ds=dataset,
                                       split=split)


