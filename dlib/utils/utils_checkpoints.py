import os
from os.path import dirname, abspath
import sys
import re
import glob
from typing import List
from copy import deepcopy

import torch
import yaml

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import fmsg
import dlib.dllogger as DLLogger
from dlib.configure import constants


__all__ = ['load_checkpoint_net', 'load_checkpoint_optimizer',
           'find_last_checkpoint', 'load_checkpoint_lr_scheduler',
           'move_state_dict_to_device',
           'save_checkpoint', 'load_loss_t', 'fn_keep_last_n_checkpoints']

_K_CHECKPOINT = 'checkpoint'
_CPU = torch.device("cpu")

_DEFAULT_CHECKPOINT = {
    constants.CHP_M: None,
    constants.CHP_O: None,
    constants.CHP_LR: None,
    constants.CHP_T: None,
    'iter': 0
}

_DEFAULT_TRACKER = {
    constants.CHP_TR: None
}

_DEFAULT_BEST_MODEL = {
    'encoder': None,
    'decoder': None,
    'classification_head': None,
    'segmentation_head': None,
    'reconstruction_head': None,
    'box_head': None
}


def move_state_dict_to_device(state_dict: dict, device):
    for k in state_dict:
        if torch.is_tensor(state_dict[k]):
            state_dict[k] = state_dict[k].to(device)

    return state_dict


def load_checkpoint_net(network, s_dict: dict, path: str = '',
                        param_key: str = 'params'):

    map_location = next(network.parameters()).device

    if s_dict is not None:
        state_dict = move_state_dict_to_device(s_dict, map_location)
    elif os.path.isfile(path):
        state_dict = torch.load(path, map_location=_CPU)[constants.CHP_M]
        state_dict = move_state_dict_to_device(state_dict, map_location)
    else:
        return 0

    if param_key in state_dict.keys():
        state_dict = state_dict[param_key]
    network.load_state_dict(state_dict, strict=True)


def load_checkpoint_optimizer(optimizer, s_dict: dict, path: str = ''):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    if s_dict is not None:
        state_dict = move_state_dict_to_device(s_dict, device)
    elif os.path.isfile(path):
        state_dict = torch.load(path, map_location=_CPU)[constants.CHP_O]
        state_dict = move_state_dict_to_device(state_dict, device)
    else:
        return 0

    optimizer.load_state_dict(state_dict)


def load_checkpoint_lr_scheduler(lr_scheduler, s_dict: dict, path: str = ''):
    if s_dict is not None:
        state_dict = s_dict
    elif os.path.isfile(path):
        state_dict = torch.load(path)[constants.CHP_LR]
    else:
        return 0

    lr_scheduler.load_state_dict(state_dict)


def load_loss_t(loss, s_t: float, path: str = ''):
    if s_t is not None:
        t = s_t
    elif os.path.isfile(path):
        t = torch.load(path)[constants.CHP_T]
    else:
        return 0

    loss.set_t(t)


def find_last_checkpoint(save_dir: str, key: str):
    file_list = glob.glob(os.path.join(save_dir, f'*_{key}.pth'))

    init_iter = 0
    init_path = ''
    if key == constants.CHP_CP:
        checkpoint = deepcopy(_DEFAULT_CHECKPOINT)
    elif key == constants.CHP_TR:
        checkpoint = deepcopy(_DEFAULT_TRACKER)
    elif key == constants.CHP_BEST_M:
        checkpoint = deepcopy(_DEFAULT_BEST_MODEL)
    else:
        raise NotImplementedError(f'key: {key}.')

    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(key), file_)
            iter_exist.append(int(iter_current[0]))

        # find the last valid (non-corrupted) checkpoint.
        iter_exist = sorted(iter_exist, reverse=True)

        for itera in iter_exist:
            init_iter = itera
            init_path = os.path.join(save_dir, f'{itera}_{key}.pth')

            try:
                checkpoint = torch.load(init_path, map_location=_CPU)
                break
            except Exception as e:
                DLLogger.log(
                    f'Failed to load checkpoint {init_path}. Error: {e}')
                os.remove(init_path)
                DLLogger.log(
                    f'Deleted checkpoint: {init_path}.')

    if init_path:
        DLLogger.log(f'Loaded checkpoint @: {init_path}')

    return init_iter, checkpoint


def fn_keep_last_n_checkpoints(save_dir: str, n: int,
                               checkpoints_health: dict, key: str):

    assert n > 0

    file_list = glob.glob(os.path.join(save_dir, f'*_{key}.pth'))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(key), file_)
            iter_exist.append(int(iter_current[0]))

        iter_exist = sorted(iter_exist, reverse=True)

        for i, itera in enumerate(iter_exist):
            _path = os.path.join(save_dir, f'{itera}_{key}.pth')

            if i >= n:
                os.remove(_path)
                checkpoints_health.pop(_path, None)
                DLLogger.log(f'Deleted extra checkpoint: {_path}.')

                continue

            try:
                if _path not in checkpoints_health:
                    torch.load(_path, map_location=_CPU)  # expensive.

                checkpoints_health[_path] = True
            except Exception as e:
                DLLogger.log(f'Failed to load checkpoint {_path}. Error: {e}')
                os.remove(_path)
                checkpoints_health.pop(_path, None)
                DLLogger.log(f'Deleted checkpoint: {_path}. Reason: {e}')
    else:
        DLLogger.log(f'no checkpoint to keep.')


def save_checkpoint(network, optimizer, lr_scheduler, loss,
                    save_dir: str,
                    current_step: int,
                    key: str):

    _network = deepcopy(network).to(_CPU).eval()

    save_filename = f'{current_step}_{key}.pth'
    save_path = os.path.join(save_dir, save_filename)

    torch.save(
        {
            constants.CHP_M: _network.state_dict(),
            constants.CHP_O: optimizer.state_dict(),
            constants.CHP_LR: lr_scheduler.state_dict(),
            constants.CHP_T: loss.get_t(),
            'iter': current_step
        },
        f=save_path
    )
    DLLogger.log(f'Saved checkpoint @ {save_path}.')


def _save_net(network, save_dir: str, iter_label: int, network_label: str):
    save_filename = f'{iter_label}_{network_label}.pth'
    save_path = os.path.join(save_dir, save_filename)

    _network = deepcopy(network).to(_CPU).eval()
    torch.save(_network.state_dict(), save_path)
    DLLogger.log(f'saved checkpoint @{network_label} @: {save_path}.')


def _save_optimizer(optimizer, save_dir: str, iter_label: int,
                    optimizer_label: str):

    save_filename = f'{iter_label}_{optimizer_label}.pth'
    save_path = os.path.join(save_dir, save_filename)
    torch.save(optimizer.state_dict(), save_path)
    DLLogger.log(f'saved checkpoint @{optimizer_label} @: {save_path}.')


def _save_lr_scheduler(scheduler, save_dir: str, iter_label: int,
                       lr_label: str):
    save_filename = f'{iter_label}_{lr_label}.pth'
    save_path = os.path.join(save_dir, save_filename)
    torch.save(scheduler.state_dict(), save_path)
    DLLogger.log(f'saved checkpoint @{lr_label} @: {save_path}.')


def _save_loss_t(l: list, save_dir: str, label: str):
    save_filename = f'{label}.yml'
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, 'w') as f:
        yaml.dump({label: l}, f)


