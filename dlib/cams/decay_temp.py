import sys
import os
from os.path import dirname, abspath

import torch
import torch.nn as nn
import numpy as np


from skimage.filters import threshold_otsu

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

__all__ = ["DecayTemp"]


class DecayTemp(object):
    def __init__(self, sl_tc_knn_t: float, sl_tc_min_t: float, sl_tc_knn: int,
                 sl_tc_knn_mode: str, sl_tc_knn_epoch_switch_uniform: int,
                 sl_tc_seed_tech: str):
        self._sl_tc_knn_mode = sl_tc_knn_mode
        self._sl_tc_knn = sl_tc_knn
        self._sl_tc_knn_t = sl_tc_knn_t
        self._sl_tc_min_t = sl_tc_min_t
        self._sl_tc_knn_epoch_switch_uniform = sl_tc_knn_epoch_switch_uniform

        self._sl_tc_seed_tech = sl_tc_seed_tech

        assert self._sl_tc_knn_t >= self._sl_tc_min_t
        assert sl_tc_knn_mode in constants.TIME_DEPENDENCY
        assert sl_tc_seed_tech in constants.SEED_TECHS

        self.decay = 0.0
        if sl_tc_knn_epoch_switch_uniform == -1:
            self.decayable = False
        else:
            self.decayable = True
            self.decay = (self._sl_tc_knn_t - self._sl_tc_min_t)
            if sl_tc_knn_epoch_switch_uniform > 0:
                self.decay = self.decay / float(sl_tc_knn_epoch_switch_uniform)
            else:
                self.decay = 0.0

        self.epoch = 0

    @property
    def sl_tc_knn_t(self) -> float:
        if not self.decayable:
            return self._sl_tc_knn_t

        val = self._sl_tc_knn_t - self.epoch * self.decay
        return max(self._sl_tc_min_t, val)

    @property
    def sl_tc_knn_mode(self) -> str:
        return self._sl_tc_knn_mode

    @property
    def sl_tc_knn(self):
        return self._sl_tc_knn

    @property
    def sl_tc_seed_tech(self) -> str:
        if not self.decayable:
            return self._sl_tc_seed_tech

        if self.epoch >= self._sl_tc_knn_epoch_switch_uniform:
            return constants.SEED_UNIFORM
        else:
            return self._sl_tc_seed_tech

    def set_epoch(self, epoch):
        assert epoch >= 0, epoch
        assert isinstance(epoch, int), type(epoch)

        self.epoch = epoch

    def get_current_status(self) -> str:
        msg = f'epoch={self.epoch},' \
              f'sl_tc_knn_t={self.sl_tc_knn_t},' \
              f'sl_tc_knn_mode={self.sl_tc_knn_mode}, ' \
              f'sl_tc_knn={self.sl_tc_knn}, ' \
              f'sl_tc_seed_tech={self.sl_tc_seed_tech}.'

        return msg

    def __str__(self):
        return f"{self.__class__.__name__}(): Decay_tmp. " \
               f"_sl_tc_knn_mode = {self._sl_tc_knn_mode}. " \
               f"_sl_tc_knn = {self._sl_tc_knn}. " \
               f"_sl_tc_knn_t = {self._sl_tc_knn_t}. " \
               f"_sl_tc_min_t = {self._sl_tc_min_t}. " \
               f"_sl_tc_knn_epoch_switch_uniform = " \
               f"{self._sl_tc_knn_epoch_switch_uniform}. " \
               f"_sl_tc_seed_tech = {self._sl_tc_seed_tech}."


def test_DecayTemp():
    tmp = DecayTemp(sl_tc_knn_t=10., sl_tc_min_t=1.0,
            sl_tc_knn=1, sl_tc_knn_mode=constants.TIME_BEFORE,
            sl_tc_knn_epoch_switch_uniform=10,
            sl_tc_seed_tech=constants.SEED_WEIGHTED)
    print(tmp)

    for e in range(20):
        tmp.set_epoch(e)
        print(f'epoch *******{e}*******')
        print(tmp.get_current_status())


if __name__ == '__main__':
    test_DecayTemp()





