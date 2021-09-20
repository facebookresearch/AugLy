#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple


def crop_bboxes_helper(
    bbox: Tuple, x1: float, y1: float, x2: float, y2: float, **kwargs
) -> Tuple:
    """
    If part of the bbox was cropped out in the x-axis, the left/right side will now be
    0/1 respectively; otherwise the fraction x1 is cut off from the left & x2 from the
    right and we renormalize with the new width. Analogous for the y-axis
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    new_w, new_h = x2 - x1, y2 - y1
    return (
        max(0, (left_factor - x1) / new_w),
        max(0, (upper_factor - y1) / new_h),
        min(1, 1 - (x2 - right_factor) / new_w),
        min(1, 1 - (y2 - lower_factor) / new_h),
    )


def hflip_bboxes_helper(bbox: Tuple, **kwargs) -> Tuple:
    """
    When the src image is horizontally flipped, the bounding box also gets horizontally
    flipped
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (1 - right_factor, upper_factor, 1 - left_factor, lower_factor)


def meme_format_bboxes_helper(
    bbox: Tuple, src_w: int, src_h: int, caption_height: int, **kwargs
) -> Tuple:
    """
    The src image is offset vertically by caption_height pixels, so we normalize that to
    get the y offset, add that to the upper & lower coordinates, & renormalize with the
    new height. The x dimension is unaffected
    """
    left_f, upper_f, right_f, lower_f = bbox
    y_off = caption_height / src_h
    new_h = 1.0 + y_off
    return left_f, (upper_f + y_off) / new_h, right_f, (lower_f + y_off) / new_h


def overlay_onto_background_image_bboxes_helper(
    bbox: Tuple, overlay_size: float, x_pos: float, y_pos: float, **kwargs
) -> Tuple:
    """
    The src image is overlaid on the dst image offset by (`x_pos`, `y_pos`) & with a
    size of `overlay_size` (all relative to the dst image dimensions). So the bounding
    box is also offset by (`x_pos`, `y_pos`) & scaled by `overlay_size`. It is also
    possible that some of the src image will be cut off, so we take the max with 0/min
    with 1 in order to crop the bbox if needed
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (
        max(0, left_factor * overlay_size + x_pos),
        max(0, upper_factor * overlay_size + y_pos),
        min(1, right_factor * overlay_size + x_pos),
        min(1, lower_factor * overlay_size + y_pos),
    )


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


def pad_square_bboxes_helper(bbox: Tuple, src_w: int, src_h: int, **kwargs) -> Tuple:
    """
    In pad_square, pad is called with w_factor & h_factor computed as follows, so we can
    use the `pad_bboxes_helper` function to transform the bbox
    """
    w_factor, h_factor = 0, 0

    if src_w < src_h:
        w_factor = (src_h - src_w) / (2 * src_w)
    else:
        h_factor = (src_w - src_h) / (2 * src_h)

    return pad_bboxes_helper(bbox, w_factor=w_factor, h_factor=h_factor)


def vflip_bboxes_helper(bbox: Tuple, **kwargs) -> Tuple:
    """
    Analogous to hflip, when the src image is vertically flipped, the bounding box also
    gets vertically flipped
    """
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (left_factor, 1 - lower_factor, right_factor, 1 - upper_factor)
