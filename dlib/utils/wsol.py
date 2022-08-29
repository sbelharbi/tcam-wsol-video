# This module shouldn't import any of our modules to avoid recursive importing.
import os
from os.path import dirname, abspath
import sys
import argparse
import textwrap
from os.path import join
import fnmatch
from pathlib import Path
import subprocess
from typing import Union

from sklearn.metrics import auc
import torch
import numpy as np
from pynvml.smi import nvidia_smi
import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['check_box_convention', 'check_scoremap_validity',
           'compute_bboxes_from_scoremaps_ext_contours']


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def compute_bboxes_from_scoremaps_ext_contours(
        scoremap: Union[list, None],
        scoremap_threshold_list,
        multi_contour_eval=False,
        bbox: Union[list, None] = None):
    """
    Use cv2.RETR_EXTERNAL mode.
    https://docs.opencv.org/4.x/d3/dc0/
    group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

        bbox: a list or None. some methods can predict a SINGLE bounding
            box without using cam. the list contains coordinates of upper
            left and lower right corners: [x1, y1, x2, y2]. x-axis: width,
            y-axi: height.

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    if scoremap is None:
        assert bbox is not None

    if scoremap is not None:
        check_scoremap_validity(scoremap)
        height, width = scoremap.shape
        scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)

        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    if scoremap is not None:
        for threshold in scoremap_threshold_list:
            boxes, number_of_box = scoremap2bbox(threshold)
            estimated_boxes_at_each_thr.append(boxes)
            number_of_box_list.append(number_of_box)
    else:
        boxes = np.array([bbox])
        number_of_box = 1
        for _ in scoremap_threshold_list:
            estimated_boxes_at_each_thr.append(boxes)
            number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list
