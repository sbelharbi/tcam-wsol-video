import os
import random
import sys
from os.path import dirname, abspath, join, basename
from typing import Optional, List, Tuple
import shutil
import pprint
from copy import deepcopy
import fnmatch

from scipy import io
import yaml
import tqdm

from PIL import Image
import numpy as np

from dlib.utils.wsol import check_box_convention

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from dlib.configure.config import get_root_wsol_dataset
from dlib.utils.reproducibility import set_seed

_NBR_VIDS = 'NBR_VIDS'
_NBR_SHOTS = 'NBR_SHOTS'
_NBR_FRAMES = 'NBR_FRAMES'
_NBR_FRAMES_W_BBOX = 'NBR_FRAMES_W_BBOX'
_NBR_BBOXES = 'NBR_BBOXes'
_NBR_VIDS_W_BBOX = 'NBR_VIDS_W_BBOX'
_NBR_SHOTS_W_BBOX = 'NBR_SHOTS_W_BBOX'

_NBRS = [_NBR_VIDS,
         _NBR_SHOTS,
         _NBR_FRAMES,
         _NBR_VIDS_W_BBOX,
         _NBR_SHOTS_W_BBOX,
         _NBR_FRAMES_W_BBOX,
         _NBR_BBOXES]

_TRAIN = 'TRAIN'
_TEST = 'TEST'

_SUBSETS = [_TRAIN, _TEST]
_CLASSES = ['aeroplane', 'bird', 'boat', 'car', 'cat', 'cow', 'dog', 'horse',
            'motorbike', 'train']

# nbr of videos to take per class for validset.
_NBR_VIDS_CL_VL = 3


def print_total(stats: dict, filelog=None):
    pp = pprint.PrettyPrinter(depth=1, stream=filelog)
    for nbr in _NBRS:
        track_subsets = {k: 0 for k in _SUBSETS}

        for cl in stats:
            for subset in stats[cl]:
                track_subsets[subset] += stats[cl][subset][nbr]

        print(f'{nbr}:', file=filelog)
        pp.pprint(track_subsets)
        print(f'Total {sum(track_subsets.values())}', file=filelog)


def get_train_test_name_vids(path_file: str) -> List[str]:
    with open(path_file, 'r') as fin:
        content = fin.readlines()

    out = [l.strip() for l in content]
    for e in out:
        assert e.startswith('0'), f'{path_file}, line: {e}'

    return out


def convert_list_videos_to_dict(lpath_fd: list) -> dict:
    out = dict()
    for fd in lpath_fd:
        out[fd] = fd

    return out


def list_name_fds(path: str) -> List[str]:
    out = []
    for item in os.listdir(path):
        if os.path.isdir(join(path, item)):
            out.append(item)

    return out


def check_if_video_has_bbox(pathvideo: str) -> bool:
    assert os.path.isdir(pathvideo), pathvideo
    out = find_files_pattern(pathvideo, '*_sticks.mat')

    return len(out) > 0


def random_select_k_vids(lvids: List[str]) -> Tuple[List[str], List[str]]:
    assert isinstance(lvids, list)

    for i in range(1000):
        random.shuffle(lvids)

    sel, leftover = lvids[:_NBR_VIDS_CL_VL], lvids[_NBR_VIDS_CL_VL:]
    return sel, leftover


def list_file_names_extension(fd_path: str, pattern_ext: str) -> List[str]:
    out = []
    content = next(os.walk(fd_path))[2]
    for item in content:
        path = join(fd_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, pattern_ext):
            out.append(item)

    out = sorted(out, reverse=False)
    return out


def get_file_names_sticks_mat(shot_path: str) -> [List[str], List[int]]:
    out = []
    content = next(os.walk(shot_path))[2]
    for item in content:
        path = join(shot_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, '*_sticks.mat'):
            out.append(item)

    out = sorted(out, reverse=False)
    nbr_boxs = []
    for item in out:
        file = join(shot_path, item)
        mat = io.loadmat(file)['coor']
        nbr_boxs.append(mat.size)
    return out, nbr_boxs


def fix_bbox(bbox: List[float], img_path: str) -> List[float]:
    # bbox: [x0, y0, x1, y1]. x@with. y@height.
    _w = bbox[2] - bbox[0]
    _h = bbox[3] - bbox[1]

    out = [v for v in bbox]
    # case 1: 'horse/data/0004/shots/055/frame0020.jpg'
    # swap x0 and x1.
    if 'horse/data/0004/shots/055/frame0020.jpg' in img_path:
        print(f'w: {_w} h: {_h} @ {img_path}')
        assert _w < 0
        out = [bbox[2], bbox[1], bbox[0], bbox[3]]

    # case 2: 'boat/data/0006/shots/026/frame0001.jpg'
    # bbox of large ship.
    # width box is larger than image. x1 > width.
    if 'boat/data/0006/shots/026/frame0001.jpg' in img_path:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if bbox[2] > w:  # bbox of large ship.
            print(f'w: {w} x1: {bbox[2]} @ {img_path}')
            # x0, y0, x1, y1. set x1 tp w - 1.
            out = [bbox[0], bbox[1], w - 1, bbox[3]]

    return out


def fix_bbox_ytov2_2(bbox: List[float], img_path: str) -> List[float]:
    # bbox: [x0, y0, x1, y1]. x@with. y@height.
    return bbox


def get_list_bbox(matfile: str) -> List[list]:
    img_path = matfile.replace('_sticks.mat', '')
    assert os.path.isfile(img_path)
    w, h = Image.open(img_path).convert('RGB').size

    mat = io.loadmat(matfile)['coor']
    out = []
    nbr_bx = mat.size
    for el in mat.reshape((nbr_bx)):
        el = numpy_relu(el)  # drop negative values.
        el = el.squeeze().tolist()
        el = fix_bbox(el, img_path)

        # double check.
        _w = el[2] - el[0]
        _h = el[3] - el[1]
        assert 0 < _w < w, f'{el}: {w}, {_w} @ {img_path}'
        assert 0 < _h < h, f'{el}: {h}, {_h} @ {img_path}'

        check_box_convention(np.array(el).reshape((1, 4)), 'x0y0x1y1')

        out.append(el)  # [x0, y0, x1, y1]. x@with. y@height. floats.

    return out


def convert_abs_path_2_rel_p(root: str, path: str) -> str:
    return path.replace(root, '').lstrip(os.sep)


def numpy_relu(v: np.ndarray) -> np.ndarray:
    return np.abs(v * (v > 0))

def convert_to_str_8_digits(x: int):
    assert isinstance(x, int), type(x)
    return str(x).zfill(8)

def check_original_ytov2_2():
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, f'{ds}-original')
    announce_msg(f'Processing {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el in _CLASSES]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    classes_id = {
        k: v for v, k in enumerate(ldirs_root)
    }

    with open(join(root_ds, "class_id.yaml"), 'w') as f:
        yaml.dump(classes_id, f)

    # get ranges file
    ranges_fd = join(root_ds, 'Ranges')
    range_cl = dict()
    for item in classes_id:
        fmt = io.loadmat(join(ranges_fd, f'ranges_{item}.mat'))['ranges']
        shot_start = fmt[0].flatten()  # np.ndarray
        shot_end = fmt[1].flatten()  # np.ndarray
        shot_video = fmt[2].flatten()  # np.ndarray

        range_cl[item] = dict()

        nbr_shots = shot_start.size

        for i in range(nbr_shots):
            s_s = shot_start[i]
            s_e = shot_end[i]
            s_v = shot_video[i]
            s_v = str(int(s_v))

            if s_v not in range_cl[item]:
                range_cl[item][s_v] = dict()

            frames = list(range(shot_start[i], shot_end[i] + 1))
            frames = [convert_to_str_8_digits(int(x)) for x in frames]
            range_cl[item][s_v][i] = frames

    # ground truth.
    gt_fd = join(root_ds, 'GroundTruth')
    gt_tr = dict()
    gt_tst = dict()

    for item in classes_id:
        tst_f = join(gt_fd, item, f'bb_gtTest_{item}.mat')
        tr_f = join(gt_fd, item, f'bb_gtTraining_{item}.mat')

        # tst
        tst_content = io.loadmat(tst_f)['bb_gtTest'][0]
        sz = tst_content.size
        for i in range(sz):
            id_, bb = tst_content[i]
            id_ = str(id_[0])
            assert id_ not in gt_tst
            gt_tst[id_] = bb.shape[0]

        # train
        train_content = io.loadmat(tr_f)['bb_gtTraining'][0]
        sz = train_content.size
        for i in range(sz):
            id_, bb = train_content[i]
            id_ = str(id_[0])
            assert id_ not in gt_tst
            gt_tr[id_] = bb.shape[0]


    nbr_bb_tst = sum([gt_tst[k] for k in gt_tst])
    nbr_bb_tr = sum([gt_tr[k] for k in gt_tr])
    total_bb = nbr_bb_tst + nbr_bb_tr

    # split train from test videos.
    tr_v = dict()
    tst_v = dict()
    ul_v = 0
    same = 0

    for item in classes_id:
        for v in range_cl[item]:
            z = 0
            t = 0
            for s in range_cl[item][v]:
                frames = deepcopy(range_cl[item][v][s])
                frames = [f'{item}{f}' for f in frames]

                intr = any([f in gt_tr for f in frames])
                intst = any([f in gt_tst for f in frames])

                if intr:
                    assert not intst
                    if item not in tr_v:
                        tr_v[item] = dict()

                    tr_v[item][v] = deepcopy(range_cl[item][v])

                if intst:
                    assert not intr
                    if item not in tst_v:
                        tst_v[item] = dict()

                    tst_v[item][v] = deepcopy(range_cl[item][v])

                if (not intr) and (not intst):
                    z += 1

                if intr and intst:
                    t += 1

            if z == len(range_cl[item][v]):
                ul_v += 1

            if t > 0:
                same += 1

    # log stats.
    with open(join(root_ds, 'original-stats.txt'), 'w') as fstats:
        msg = f'dataset: {ds} \n'
        msg += f'nbr videos that are not labeled: {ul_v} \n'
        msg += f'nbr SAME videos in tr and tst: {same} \n'
        msg += f'nbr bb: {total_bb} (train: {nbr_bb_tr}, tst: {nbr_bb_tst}) \n'
        msg += f'nbr labeled frames in trainset: {len(gt_tr)} \n'
        msg += f'nbr labeled frames in tstset: {len(gt_tst)} \n'
        stats_tr_v = 0
        stats_ts_v = 0
        stats_tr_s = 0
        stats_ts_s = 0
        stats_tr_f = 0
        stats_ts_f = 0

        for cl in classes_id:
            stats_tr_v += len(tr_v[cl])
            stats_ts_v += len(tst_v[cl])

            for v in tr_v[cl]:
                stats_tr_s += len(tr_v[cl][v])
                for s in tr_v[cl][v]:
                    stats_tr_f += len(tr_v[cl][v][s])

            for v in tst_v[cl]:
                stats_ts_s += len(tst_v[cl][v])
                for s in tst_v[cl][v]:
                    stats_ts_f += len(tst_v[cl][v][s])

        msg += f'train: \n nbr videos: {stats_tr_v} \n ' \
               f'nbr shots: {stats_tr_s} \n '\
               f'nbr frames: {stats_tr_f} \n '

        msg += f'test: \n nbr videos: {stats_ts_v} \n ' \
               f'nbr shots: {stats_ts_s} \n ' \
               f'nbr frames: {stats_ts_f} \n'

        fstats.write(msg)
        print(msg)

def convert_to_str_n_digits(x: int, n: int):
    assert isinstance(x, int), type(x)
    return str(x).zfill(n)

def convert_to_std(cl: str, video: int, shot: int, frame: str) -> str:
    v = convert_to_str_n_digits(video, 4)
    s = convert_to_str_n_digits(shot, 6)
    r_path = f'{cl}/video-{v}/shot-{s}/{frame}.jpg'
    return r_path


def copy_src_des_frms(src: list, src_root: str, dest: list, dest_root: str):
    assert len(src) == len(dest)

    for s, d in zip(src, dest):
        _d = join(dest_root, d)
        _s = join(src_root, s)
        os.makedirs(dirname(_d), exist_ok=True)
        shutil.copy(_s, _d)
        print(f'copied: {s} -> {d}')


def process_ytov2_2():
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, f'{ds}-original')
    announce_msg(f'Processing {ds} @ {root_ds}')
    new_root_ds = join(root, ds)

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el in _CLASSES]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    classes_id = {
        k: v for v, k in enumerate(ldirs_root)
    }

    with open(join(new_root_ds, "class_id.yaml"), 'w') as f:
        yaml.dump(classes_id, f)

    folds_dir = join(root_dir, constants.RELATIVE_META_ROOT, ds)
    os.makedirs(folds_dir, exist_ok=True)
    with open(join(folds_dir, "class_id.yaml"), 'w') as f:
        yaml.dump(classes_id, f)


    # get ranges file
    ranges_fd = join(root_ds, 'Ranges')
    range_cl = dict()
    all_data = dict()
    all_short_to_long_id = dict()

    for item in classes_id:
        fmt = io.loadmat(join(ranges_fd, f'ranges_{item}.mat'))['ranges']
        shot_start = fmt[0].flatten()  # np.ndarray
        shot_end = fmt[1].flatten()  # np.ndarray
        shot_video = fmt[2].flatten()  # np.ndarray

        range_cl[item] = dict()
        all_data[item] = dict()

        nbr_shots = shot_start.size

        for i in range(nbr_shots):
            s_s = shot_start[i]
            s_e = shot_end[i]
            s_v = shot_video[i]
            s_v = str(int(s_v))

            if s_v not in range_cl[item]:
                range_cl[item][s_v] = dict()
                all_data[item][s_v] = dict()

            frames = list(range(shot_start[i], shot_end[i] + 1))
            frames = [convert_to_str_8_digits(int(x)) for x in frames]
            for f in frames:
                _id = f'{item}{f}'
                # todo: there is an issue with shot containing frames 38316
                #  for class horse. shot ends: 38465, 38465 was repeated twice.
                if _id in all_short_to_long_id:
                    print(f'error in data: {_id}')

                # assert _id not in all_short_to_long_id, _id
                all_short_to_long_id[_id] = convert_to_std(
                    item, int(shot_video[i]), i, f)

            file_frames = [join(root_ds, item, f'{x}.jpg') for x in frames]
            error = zip(file_frames,
                        [os.path.isfile(x) for x in file_frames])
            assert all([os.path.isfile(x) for x in file_frames]), list(error)

            src_fr = [join(item, f'{x}.jpg') for x in frames]
            des_fr = [convert_to_std(item, int(shot_video[i]), i, x) for x in
                      frames]

            range_cl[item][s_v][i] = frames
            all_data[item][s_v][i] = des_fr
            # todo: uncomment
            copy_src_des_frms(src=src_fr, src_root=root_ds, dest=des_fr,
                              dest_root=new_root_ds)

    # ground truth.
    gt_fd = join(root_ds, 'GroundTruth')
    gt_tr = dict()
    gt_tst = dict()
    info_tst = []
    info_tr = []

    for item in classes_id:
        tst_f = join(gt_fd, item, f'bb_gtTest_{item}.mat')
        tr_f = join(gt_fd, item, f'bb_gtTraining_{item}.mat')

        # tst
        tst_content = io.loadmat(tst_f)['bb_gtTest'][0]
        sz = tst_content.size
        for i in range(sz):
            id_, bb = tst_content[i]
            id_ = str(id_[0])
            id_final = all_short_to_long_id[id_]
            assert id_final not in gt_tst
            gt_tst[id_final] = process_bb_ytov2_2(
                bb, join(root_ds, get_path_from_id(id_, item)))
            l_bbox = gt_tst[id_final]
            for bb in l_bbox:
                info_tst.append([id_final, classes_id[item], bb])

        # train
        train_content = io.loadmat(tr_f)['bb_gtTraining'][0]
        sz = train_content.size
        for i in range(sz):
            id_, bb = train_content[i]

            id_ = str(id_[0])
            id_final = all_short_to_long_id[id_]
            assert id_final not in gt_tst
            gt_tr[id_final] = process_bb_ytov2_2(
                bb, join(root_ds, get_path_from_id(id_, item)))
            l_bbox = gt_tr[id_final]
            for bb in l_bbox:
                info_tr.append([id_final, classes_id[item], bb])

    # split train from test videos.
    tr_v = dict()
    tst_v = dict()

    for item in classes_id:
        for v in range_cl[item]:
            for s in range_cl[item][v]:
                frames = deepcopy(range_cl[item][v][s])
                frames = [f'{item}{f}' for f in frames]
                frames = [all_short_to_long_id[f] for f in frames]

                intr = any([f in gt_tr for f in frames])
                intst = any([f in gt_tst for f in frames])

                if intr:
                    assert not intst
                    if item not in tr_v:
                        tr_v[item] = dict()

                    tr_v[item][v] = deepcopy(range_cl[item][v])

                if intst:
                    assert not intr
                    if item not in tst_v:
                        tst_v[item] = dict()

                    tst_v[item][v] = deepcopy(range_cl[item][v])

    # full test
    info_tst_full = []
    for cl in classes_id:
        for v in tst_v[cl]:
            for s in tst_v[cl][v]:
                frames = all_data[cl][v][s]  # std format.
                frames_short_id = tst_v[cl][v][s]
                frames_short_id = [f'{cl}{f}' for f in frames_short_id]

                assert len(frames) == len(frames_short_id)

                for flong, fshort in zip(frames, frames_short_id):
                    id_final = flong
                    if fshort in gt_tst:
                        l_bbox = gt_tst[id_final]
                        for bb in l_bbox:
                            info_tst_full.append([id_final, classes_id[cl], bb])
                    else:
                        zero_bbox = [0., 0., 0., 0.]  # null bbox.
                        info_tst_full.append(
                            [id_final, classes_id[cl], zero_bbox])

    # build test from frames only.
    build_test_ytov2_2(info_tst)
    # build test for all frames.
    build_video_demo_test_ytov2_2(info_tst_full)

    # train/valid sets
    _tr_v, _vl_v = split_train_vl_ytov2_2(tr_v, classes_id)
    valid_info = build_vl_info_ytov2_2(classes_id, _vl_v, gt_tr, all_data)
    train_info = build_tr_info_ytov2_2(classes_id, _tr_v, all_data)
    build_train_valid_ytov2_2(valid_info, train_info)

    with open(join(new_root_ds, 'stats-train-valid.txt'), 'w') as fstats:
        msg = f'dataset: {ds} \n'
        stats_tr_v = 0
        stats_vl_v = 0
        stats_tr_s = 0
        stats_vl_s = 0
        stats_tr_f = 0
        stats_vl_f = 0

        for cl in classes_id:
            stats_tr_v += len(_tr_v[cl])
            stats_vl_v += len(_vl_v[cl])

            for v in _tr_v[cl]:
                stats_tr_s += len(_tr_v[cl][v])
                for s in _tr_v[cl][v]:
                    stats_tr_f += len(_tr_v[cl][v][s])

            for v in _vl_v[cl]:
                stats_vl_s += len(_vl_v[cl][v])
                for s in _vl_v[cl][v]:
                    stats_vl_f += len(_vl_v[cl][v][s])

        msg += f'train: \n nbr videos: {stats_tr_v} \n ' \
               f'nbr shots: {stats_tr_s} \n ' \
               f'nbr frames: {stats_tr_f} \n '

        msg += f'valid: \n nbr videos: {stats_vl_v} \n ' \
               f'nbr shots: {stats_vl_s} \n ' \
               f'nbr frames: {stats_vl_f} \n'

        fstats.write(msg)
        print(msg)


def get_path_from_id(_id: str, cl_name: str) -> str:
    assert cl_name in _id, f"{cl_name}: {_id}"
    _id = _id.replace(cl_name, '')
    return f'{cl_name}/{_id}.jpg'

def process_bb_ytov2_2(bbox: np.ndarray, img_path: str) -> list:
    assert os.path.isfile(img_path), img_path

    w, h = Image.open(img_path).convert('RGB').size
    out = []
    nbr_bx = bbox.shape[0]
    for i in range(nbr_bx):
        el = bbox[i]
        el = numpy_relu(el)  # drop negative values.
        el = el.squeeze().tolist()
        el = fix_bbox_ytov2_2(el, img_path)

        # double check.
        _w = el[2] - el[0]
        _h = el[3] - el[1]
        assert 0 < _w < w, f'{el}: {w}, {_w} @ {img_path}'
        assert 0 < _h < h, f'{el}: {h}, {_h} @ {img_path}'

        check_box_convention(np.array(el).reshape((1, 4)), 'x0y0x1y1')

        out.append(el)  # [x0, y0, x1, y1]. x@with. y@height. floats.

    return out

def build_test_ytov2_2(info: list):
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET} set for {ds} @ {root_ds}')

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET)
    os.makedirs(outfolds, exist_ok=True)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')


def build_video_demo_test_ytov2_2(info: list):
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET_VIDEO_DEMO} set for {ds} @'
                 f' {root_ds}')

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET_VIDEO_DEMO)
    os.makedirs(outfolds, exist_ok=True)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')


def build_train_valid_ytov2_2(info_valid, info_train):
    """
    Build train and valid set for YTO v2.2.
    the original trainset videos has some videos with some shots that have
    bounding boxes.

    the original trainset of VIDEOS is split into two sets: videos for train,
    and videos for validation. videos in validset has bounding boxes
    localization.

    for each class, we take _NBR_VIDS_CL_VL videos that have bbox for
    validset. from these selected videos for validset, ONLY frames that have
    bbox are used. frames without bbox labels are discarded [not used].


    all other videos that are not considered for validset are used for
    trainset. we index shots instead of frames. when using dataloader,
    it will sample over shots. internally, we randomly select frame from the
    selected shot.
    """
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TRAINSET} and {constants.VALIDSET} '
                 f'set for {ds} @ {root_ds}')

    # validset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.VALIDSET)
    os.makedirs(outfolds, exist_ok=True)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info_valid:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')

    # trainset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TRAINSET)
    os.makedirs(outfolds, exist_ok=True)

    # shuffle train shots.
    set_seed(0)
    for i in range(10000):
        random.shuffle(info_train)

    # image_ids.txt
    # <path>
    # path/to
    # path/to
    # path/to
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to,0
    # path/to,1
    # path/to,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    with open(join(outfolds, 'image_sizes.txt'), 'w'):
        pass

    # localization.txt  #empty.
    with open(join(outfolds, 'localization.txt'), 'w'):
        pass


def build_test_ytov1_0():
    """
    Build test set for YTO v1.0 only over frames with bbox. used for evaluated.
    """
    ds = constants.YTOV1
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET} set for {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET)
    os.makedirs(outfolds, exist_ok=True)
    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.load(f)

    holder = dict()
    info = []
    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        h = ['test.txt']
        paths = []
        for el in h:
            pathx = join(root_ds, cl, f'sets/{el}')
            assert os.path.isfile(pathx)

            lfds = get_train_test_name_vids(pathx)
            lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

            for e in lfds:
                assert os.path.isdir(e), e

            paths.append(lfds)

        holder[cl] = {
            _TEST: dict()
        }

        test_vids_fds = paths[0]
        holder[cl][_TEST]: dict = convert_list_videos_to_dict(test_vids_fds)

        # 2. shots + frames.

        subset = _TEST
        for vid in holder[cl][subset]:
            vids_has_bbox = False
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)
                    if os.path.isfile(path) and fnmatch.fnmatch(path,
                                                                '*_sticks.mat'):

                        img_file = path.replace('_sticks.mat', '')
                        assert os.path.isfile(img_file), img_file

                        identifier = convert_abs_path_2_rel_p(root_ds, img_file)
                        l_bbox = get_list_bbox(path)
                        for bb in l_bbox:
                            info.append([identifier, classes_id[cl], bb])

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')


def build_video_demo_test_ytov1_0():
    """
    Build test set over YTO v1.0 using all frames. useful for demos.
    """
    ds = constants.YTOV1
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TESTSET_VIDEO_DEMO} set for {ds} @'
                 f' {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TESTSET_VIDEO_DEMO)
    os.makedirs(outfolds, exist_ok=True)
    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.load(f)

    holder = dict()
    info = []
    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        h = ['test.txt']
        paths = []
        for el in h:
            pathx = join(root_ds, cl, f'sets/{el}')
            assert os.path.isfile(pathx)

            lfds = get_train_test_name_vids(pathx)
            lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

            for e in lfds:
                assert os.path.isdir(e), e

            paths.append(lfds)

        holder[cl] = {
            _TEST: dict()
        }

        test_vids_fds = paths[0]
        holder[cl][_TEST]: dict = convert_list_videos_to_dict(test_vids_fds)

        # 2. shots + frames.

        subset = _TEST
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)

                    if os.path.isfile(path) and fnmatch.fnmatch(path, '*.jpg'):
                        identifier = convert_abs_path_2_rel_p(root_ds, path)
                        path_bbox = path + '_sticks.mat'
                        if os.path.isfile(path_bbox):
                            l_bbox = get_list_bbox(path_bbox)
                            for bb in l_bbox:
                                info.append([identifier, classes_id[cl], bb])
                        else:
                            zero_bbox = [0., 0., 0., 0.]  # null bbox.
                            info.append([identifier, classes_id[cl], zero_bbox])

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')

def split_train_vl_ytov2_2(tr_v: dict, classes_id: dict):
    _tr_v, _vl_v = dict(), dict()

    for cl in classes_id:
        lv = list(tr_v[cl])
        _vl_v[cl] = dict()
        _tr_v[cl] = dict()

        for v in lv[:_NBR_VIDS_CL_VL]:
            _vl_v[cl][v] = deepcopy(tr_v[cl][v])

        for v in lv[_NBR_VIDS_CL_VL:]:
            _tr_v[cl][v] = deepcopy(tr_v[cl][v])

    return _tr_v, _vl_v


def build_vl_info_ytov2_2(classes_id: dict, l_v: dict, gt: dict,
                          all_data: dict):
    info = []
    for cl in classes_id:
        for v in l_v[cl]:
            for s in l_v[cl][v]:
                frames = all_data[cl][v][s]  # std format.
                # frames_short_id = l_v[cl][v][s]
                # frames_short_id = [f'{cl}{f}' for f in frames_short_id]

                # assert len(frames) == len(frames_short_id)

                for f in frames:
                    if f in gt:
                        id_final = f
                        l_bbox = gt[id_final]
                        for bb in l_bbox:
                            info.append([id_final, classes_id[cl], bb])

    return info


def build_tr_info_ytov2_2(classes_id: dict, l_v: dict, all_data: dict):
    info = []
    for cl in classes_id:
        for v in l_v[cl]:
            for s in l_v[cl][v]:
                frames = all_data[cl][v][s]  # std format.

                f0 = frames[0]
                id_final = dirname(f0).lstrip(os.sep)
                info.append([id_final, classes_id[cl], None])

    return info


def build_train_valid_ytov1_0():
    """
    Build train and valid set for YTO v1.0.
    the original trainset videos has some videos with some shots that have
    bounding boxes.

    the original trainset of VIDEOS is split into two sets: videos for train,
    and videos for validation. videos in validset has bounding boxes
    localization.

    for each class, we take _NBR_VIDS_CL_VL videos that have bbox for
    validset. from these selected videos for validset, ONLY frames that have
    bbox are used. frames without bbox labels are discarded [not used].


    all other videos that are not considered for validset are used for
    trainset. we index shots instead of frames. when using dataloader,
    it will sample over shots. internally, we randomly select frame from the
    selected shot.
    """
    ds = constants.YTOV1
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    announce_msg(f'Building {constants.TRAINSET} and {constants.VALIDSET} '
                 f'set for {ds} @ {root_ds}')

    ldirs_root = next(os.walk(root_ds))[1]
    ldirs_root = [el for el in ldirs_root if el not in ['vo-release', 'code']]
    assert len(ldirs_root) == 10, f"found: {len(ldirs_root)} dirs."
    ldirs_root = sorted(ldirs_root, reverse=False)

    with open(join(root_ds, "class_id.yaml"), 'r') as f:
        classes_id = yaml.load(f)

    holder = dict()
    info_valid = []
    info_train = []
    nbr_vids = {
        constants.TRAINSET: 0,
        constants.VALIDSET: 0
    }
    seed = 0

    for cl in tqdm.tqdm(classes_id, ncols=80,
                        total=len(list(classes_id.keys()))):
        # 1. vids.
        pathx = join(root_ds, cl, 'sets/train.txt')
        assert os.path.isfile(pathx)

        lfds = get_train_test_name_vids(pathx)
        lfds = [join(root_ds, cl, 'data', fd) for fd in lfds]

        labeld = []
        unlabeled = []
        for e in lfds:
            assert os.path.isdir(e), e
            if check_if_video_has_bbox(e):
                labeld.append(e)
            else:
                unlabeled.append(e)

        labeld = sorted(labeld, reverse=False)
        unlabeled = sorted(unlabeled, reverse=False)

        set_seed(seed)

        selected, leftover = random_select_k_vids(labeld)
        unlabeled += leftover

        nbr_vids[constants.VALIDSET] += len(selected)
        nbr_vids[constants.TRAINSET] += len(unlabeled)

        seed += 1

        holder[cl] = {
            constants.TRAINSET: dict(),
            constants.VALIDSET: dict()
        }

        holder[cl][constants.TRAINSET]: dict = convert_list_videos_to_dict(
            unlabeled)
        holder[cl][constants.VALIDSET]: dict = convert_list_videos_to_dict(
            selected)

        # 2. shots + frames.
        subset = constants.VALIDSET
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            # frames
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                content = next(os.walk(shot))[2]
                for item in content:
                    path = join(shot, item)
                    if os.path.isfile(path) and fnmatch.fnmatch(path,
                                                                '*_sticks.mat'):

                        img_file = path.replace('_sticks.mat', '')
                        assert os.path.isfile(img_file), img_file

                        identifier = convert_abs_path_2_rel_p(root_ds, img_file)
                        l_bbox = get_list_bbox(path)
                        for bb in l_bbox:
                            info_valid.append([identifier, classes_id[cl], bb])

        subset = constants.TRAINSET
        for vid in holder[cl][subset]:
            # shots
            list_dirs_shots = list_name_fds(join(vid, 'shots'))
            for shot in list_dirs_shots:
                shot = join(vid, 'shots', shot)
                l_frames = list_file_names_extension(shot,
                                                     pattern_ext='*.jpg')
                # add only shots that have frames.
                # some shots are empty.
                if len(l_frames):
                    identifier = convert_abs_path_2_rel_p(root_ds, shot)
                    info_train.append([identifier, classes_id[cl], None])

    # validset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.VALIDSET)
    os.makedirs(outfolds, exist_ok=True)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_valid:
            if iden not in idens:
                idens.append(iden)
                img_path = join(root_ds, iden)
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                fids.write(f'{iden},{w},{h}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303
    with open(join(outfolds, 'localization.txt'), 'w') as fids:
        for iden, cl, bbox in info_valid:
            fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')

    # trainset -----------------------------------------------------------------
    outfolds = join(root_dir, constants.RELATIVE_META_ROOT, ds,
                    constants.TRAINSET)
    os.makedirs(outfolds, exist_ok=True)

    # shuffle train shots.
    set_seed(0)
    for i in range(10000):
        random.shuffle(info_train)

    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg
    with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1
    with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
        idens = []
        for iden, cl, bbox in info_train:
            if iden not in idens:
                idens.append(iden)
                fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300
    with open(join(outfolds, 'image_sizes.txt'), 'w'):
        pass

    # localization.txt  #empty.
    with open(join(outfolds, 'localization.txt'), 'w'):
        pass

    with open(join(root_ds, 'stats-train-valid.txt'), 'w') as fstats:
        pp = pprint.PrettyPrinter(depth=3, stream=fstats)
        pp.pprint(nbr_vids)


def compress_only_needed_frames_ytov2_2():
    """
    In a different COPY of the dataset, we keep ONLY the NEEDED frames.
    For the trainset, we keep ALL frames from every included shot.
    For valid and test sets, we only keep used frames.
    This yields a COPY of the dataset with only necessary files.
    This could be useful on servers with limit quota in terms of number of
    files.

    COPY ONLY JPG FILES.
    """
    ds = constants.YTOV22
    root = get_root_wsol_dataset()
    root_ds = join(root, ds)
    outputfd = join(root, 'compressed', ds)
    os.makedirs(outputfd, exist_ok=True)
    announce_msg(f'Compressing {ds} dataset @ {root_ds}')

    # get all train frames.
    folds = {
        constants.TRAINSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                 constants.TRAINSET),
        constants.VALIDSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                 constants.VALIDSET),
        constants.TESTSET: join(root_dir, constants.RELATIVE_META_ROOT, ds,
                                constants.TESTSET)
    }

    with open(join(folds[constants.TRAINSET], 'image_ids.txt'), 'r') as fin:
        train_shots = fin.readlines()

    train_frames = []
    for shot in train_shots:
        shot = join(root_ds, shot.strip())
        assert os.path.isdir(shot)
        l_frames = list_file_names_extension(shot,
                                             pattern_ext='*.jpg')
        # add only shots that have frames.
        # some shots are empty.
        if len(l_frames):
            l_frames = [join(shot, frame) for frame in l_frames]
            train_frames += l_frames

    # train and test
    with open(join(folds[constants.VALIDSET], 'image_ids.txt'), 'r') as fin:
        valid_frames = fin.readlines()
        valid_frames = [join(root_ds, f.strip()) for f in valid_frames]
        assert all([os.path.isfile(f) for f in valid_frames])

    with open(join(folds[constants.TESTSET], 'image_ids.txt'), 'r') as fin:
        test_frames = fin.readlines()
        test_frames = [join(root_ds, f.strip()) for f in test_frames]
        assert all([os.path.isfile(f) for f in test_frames])

    necessary_frames = train_frames + valid_frames + test_frames
    print(f'@{ds}: Number of necessary frames {len(necessary_frames)}')

    print('Making a copy...')
    for frame in tqdm.tqdm(necessary_frames, ncols=80, total=len(
            necessary_frames)):
        source = frame
        dest = source.replace(root_ds, outputfd)
        os.makedirs(dirname(dest), exist_ok=True)
        shutil.copy(source, dest)


if __name__ == '__main__':

    # ==========================================================================
    #                                  YTO V2.2
    # ==========================================================================

    # 1. get stats from original dataset.
    # check_original_ytov2_2()

    # 2.
    # - copy dataset into different folder with paths: video/shot/frames
    # - build test
    # - build full test (demo)
    # - build train/valid

    process_ytov2_2()

    # compress dataset into a copy and keep only necessary frames (train:
    # full, valid/test: only labeled frames)
    compress_only_needed_frames_ytov2_2()

    print('Done.')

