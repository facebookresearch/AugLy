#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple


def pad_bboxes_helper(bbox: Tuple, w_factor: float, h_factor: float, **kwargs) -> Tuple:
    """
    The src image is padded horizontally with w_factor * src_w, so the bbox gets shifted
    over by w_factor and then renormalized over the new width. Vertical padding is
    analogous
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    new_w = 1 + 2 * w_factor
    new_h = 1 + 2 * h_factor
    return (
        (left_factor + w_factor) / new_w,
        (upper_factor + h_factor) / new_h,
        (right_factor + w_factor) / new_w,
        (lower_factor + h_factor) / new_h,
    )
