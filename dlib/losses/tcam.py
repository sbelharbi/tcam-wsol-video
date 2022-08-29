import sys
from os.path import dirname, abspath
from typing import Tuple

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.core import ElementaryLoss
from dlib.losses.entropy import Entropy
from dlib.crf.dense_crf_loss import DenseCRFLoss
from dlib.crf.color_dense_crf_loss import ColorDenseCRFLoss


__all__ = [
    'SelfLearningTcams',
    'ConRanFieldTcams',
    'EntropyTcams',
    'MaxSizePositiveTcams',
    'RgbJointConRanFieldTcams',
    'BgSizeGreatSizeFgTcams',
    'FgSizeTcams',
    'EmptyOutsideBboxTcams'
]


def group_ordered_frames(seq_iter: torch.Tensor,
                         frm_iter: torch.Tensor) -> list:
    uniq_seq = torch.unique(seq_iter, sorted=True, return_inverse=False,
                            return_counts=False)
    out = []
    for s in uniq_seq:
        idx = torch.nonzero(seq_iter == s, as_tuple=False).view(-1, )
        h = [[i, frm_iter[i]] for i in idx]
        ordered = sorted(h, key=lambda x: x[1], reverse=False)
        o_idx = [x[0] for x in ordered]

        out.append(o_idx)

    return out


class SelfLearningTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningTcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(SelfLearningTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldTcams, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(ConRanFieldTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyTcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(EntropyTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class RgbJointConRanFieldTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(RgbJointConRanFieldTcams, self).__init__(**kwargs)

        self.loss = ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor).to(self._device)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(RgbJointConRanFieldTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        ordered_frames = group_ordered_frames(seq_iter, frm_iter)

        c = 0.
        loss = self._zero
        for item in ordered_frames:
            if len(item) < 2:
                continue

            p_imgs, p_cams = self.pair_samples(
                o_idx=item, imgs=raw_img, prob_cams=fcams_n)
            loss = loss + self.loss(images=p_imgs, segmentations=p_cams)
            c += 1.
        # todo: check c is non zero.
        return loss / c

    @staticmethod
    def pair_samples(o_idx: list,
                     imgs: torch.Tensor,
                     prob_cams: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:

        assert imgs.ndim == 4, imgs.ndim
        assert imgs.shape[1] == 3, imgs.shape[1]
        assert prob_cams.ndim == 4, prob_cams.ndim
        assert len(o_idx) > 1, len(o_idx)

        out_img = None
        out_prob_cams = None
        for i in o_idx:
            tmp_img = imgs[i].unsqueeze(0)
            tmp_prob_cams = prob_cams[i].unsqueeze(0)

            if out_img is None:
                out_img = tmp_img
                out_prob_cams = tmp_prob_cams
            else:
                # cat width.
                out_img = torch.cat((out_img, tmp_img), dim=3)
                out_prob_cams = torch.cat((out_prob_cams, tmp_prob_cams), dim=3)

        return out_img, out_prob_cams


class MaxSizePositiveTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveTcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(MaxSizePositiveTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams_n.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1. / 2.)


class BgSizeGreatSizeFgTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(BgSizeGreatSizeFgTcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(BgSizeGreatSizeFgTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]

        n = fcams_n.shape[0]
        bg = fcams_n[:, 0].view(n, -1).sum(dim=-1).view(-1, )
        fg = fcams_n[:, 1].view(n, -1).sum(dim=-1).view(-1, )
        diff = bg - fg
        loss = self.elb(-diff)

        return self.lambda_ * loss


class FgSizeTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(FgSizeTcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)
        self.eps = 0.0
        self.eps_already_set = False

    def set_eps(self, eps: float):
        assert eps >= 0, eps
        assert isinstance(eps, float), type(eps)
        self.eps = eps

        self.eps_already_set = True

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(FgSizeTcams, self).forward(epoch=epoch)

        assert self.eps_already_set, 'set it first.'

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]

        n, _, h, w = fcams_n.shape
        fg = fcams_n[:, 1].view(n, -1).sum(dim=-1).view(-1, ) / float(h * w)
        diff1 = fg_size - self.eps - fg
        loss = self.elb(diff1)
        diff2 = fg - fg_size - self.eps
        loss = loss + self.elb(diff2)

        return self.lambda_ * loss / 2.


class EmptyOutsideBboxTcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EmptyOutsideBboxTcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(EmptyOutsideBboxTcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]
        assert msk_bbox.ndim == 4, msk_bbox.ndim
        assert msk_bbox.shape[1] == 1, msk_bbox.shape[1]
        assert msk_bbox.shape[0] == fcams_n.shape[0], f'{msk_bbox.shape}, ' \
                                                      f'{fcams_n.shape}'
        assert msk_bbox.shape[2:] == fcams_n.shape[2:], f'{msk_bbox.shape}, ' \
                                                        f'{fcams_n.shape}'

        n = fcams_n.shape[0]
        out = fcams_n[:, 1].unsqueeze(1) * (1. - msk_bbox)  # b, 1, h, w
        area = out.view(n, -1).sum(dim=-1).view(-1, )
        loss = self.elb(area)

        return self.lambda_ * loss