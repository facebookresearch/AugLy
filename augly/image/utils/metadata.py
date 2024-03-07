#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from augly.image import intensity as imintensity
from augly.image.utils import bboxes as imbboxes
from PIL import Image


def normalize_bbox(bbox: Tuple, bbox_format: str, src_w: int, src_h: int) -> Tuple:
    if bbox_format == "pascal_voc_norm":
        return bbox
    elif bbox_format == "pascal_voc":
        # (left, upper, right, lower) -> normalize
        left, upper, right, lower = bbox
        return (left / src_w, upper / src_h, right / src_w, lower / src_h)
    elif bbox_format == "coco":
        # (left, upper, w, h) -> add left & upper to w & h, normalize
        left, upper, w, h = bbox
        return (left / src_w, upper / src_h, (left + w) / src_w, (upper + h) / src_h)
    else:
        # (x_center_norm, y_center_norm, w_norm, h_norm) -> compute left & upper
        x_center_norm, y_center_norm, w_norm, h_norm = bbox
        left_norm = x_center_norm - w_norm / 2
        upper_norm = y_center_norm - h_norm / 2
        return (left_norm, upper_norm, left_norm + w_norm, upper_norm + h_norm)


def validate_and_normalize_bboxes(
    bboxes: List[Tuple], bbox_format: str, src_w: int, src_h: int
) -> List[Tuple]:
    norm_bboxes = []
    for bbox in bboxes:
        assert len(bbox) == 4 and all(
            isinstance(x, (float, int)) for x in bbox
        ), f"Bounding boxes must be tuples of 4 floats; {bbox} is invalid"

        norm_bboxes.append(normalize_bbox(bbox, bbox_format, src_w, src_h))

        assert (
            0 <= norm_bboxes[-1][0] <= norm_bboxes[-1][2] <= 1
            and 0 <= norm_bboxes[-1][1] <= norm_bboxes[-1][3] <= 1
        ), f"Bounding box {bbox} is invalid or is not in {bbox_format} format"

    return norm_bboxes


def convert_bboxes(
    transformed_norm_bboxes: List[Optional[Tuple]],
    bboxes: List[Tuple],
    bbox_format: str,
    aug_w: int,
    aug_h: int,
) -> None:
    if bbox_format == "pascal_voc_norm":
        return

    for i, bbox in enumerate(transformed_norm_bboxes):
        if bbox is None:
            continue

        left_norm, upper_norm, right_norm, lower_norm = bbox
        if bbox_format == "pascal_voc":
            # denormalize -> (left, upper, right, lower)
            bboxes[i] = (
                left_norm * aug_w,
                upper_norm * aug_h,
                right_norm * aug_w,
                lower_norm * aug_h,
            )
        elif bbox_format == "coco":
            # denormalize, get w & h -> (left, upper, w, h)
            left, upper = left_norm * aug_w, upper_norm * aug_h
            right, lower = right_norm * aug_w, lower_norm * aug_h
            bboxes[i] = (left, upper, right - left, lower - upper)
        else:
            # compute x & y center -> (x_center_norm, y_center_norm, w_norm, h_norm)
            w_norm, h_norm = right_norm - left_norm, lower_norm - upper_norm
            x_center_norm = left_norm + w_norm / 2
            y_center_norm = upper_norm + h_norm / 2
            bboxes[i] = (x_center_norm, y_center_norm, w_norm, h_norm)


def check_for_gone_bboxes(transformed_bboxes: List[Tuple]) -> List[Optional[Tuple]]:
    """
    When a bounding box is cropped out of the image or something is overlaid
    which obfuscates it, we consider the bbox to no longer be visible/valid, so
    we will return it as None
    """
    checked_bboxes = []
    for transformed_bbox in transformed_bboxes:
        left_factor, upper_factor, right_factor, lower_factor = transformed_bbox
        checked_bboxes.append(
            None
            if left_factor >= right_factor or upper_factor >= lower_factor
            else transformed_bbox
        )
    return checked_bboxes


def transform_bboxes(
    function_name: str,
    image: Image.Image,
    aug_image: Image.Image,
    dst_bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
    bboxes_helper_func: Optional[Callable] = None,
    **kwargs,
) -> None:
    if dst_bboxes is None:
        return

    assert bbox_format is not None and bbox_format in [
        "pascal_voc",
        "pascal_voc_norm",
        "coco",
        "yolo",
    ], "bbox_format must be specified if bboxes are passed in and must be a supported format"

    src_w, src_h = image.size
    aug_w, aug_h = aug_image.size
    norm_bboxes = validate_and_normalize_bboxes(dst_bboxes, bbox_format, src_w, src_h)

    if bboxes_helper_func is None:
        bboxes_helper_func = getattr(
            imbboxes, f"{function_name}_bboxes_helper", lambda bbox, **_: bbox
        )

    func_kwargs = deepcopy(kwargs)
    func_kwargs.pop("src_bboxes", None)
    transformed_norm_bboxes = [
        bboxes_helper_func(bbox=bbox, src_w=src_w, src_h=src_h, **func_kwargs)
        for bbox in norm_bboxes
    ]

    transformed_norm_bboxes = check_for_gone_bboxes(transformed_norm_bboxes)
    convert_bboxes(transformed_norm_bboxes, dst_bboxes, bbox_format, aug_w, aug_h)


def get_func_kwargs(
    metadata: Optional[List[Dict[str, Any]]],
    local_kwargs: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    if metadata is None:
        return {}

    bboxes = local_kwargs.pop("bboxes", None)
    bboxes = bboxes if len(metadata) == 0 else metadata[-1].get("dst_bboxes", None)

    func_kwargs = deepcopy(local_kwargs)
    func_kwargs.pop("metadata")

    if bboxes is not None:
        func_kwargs["src_bboxes"] = deepcopy(bboxes)
        func_kwargs["dst_bboxes"] = deepcopy(bboxes)
    func_kwargs.update(**deepcopy(kwargs))

    return func_kwargs


def get_metadata(
    metadata: Optional[List[Dict[str, Any]]],
    function_name: str,
    image: Optional[Image.Image] = None,
    aug_image: Optional[Image.Image] = None,
    bboxes: Optional[Tuple] = None,
    bboxes_helper_func: Optional[Callable] = None,
    **kwargs,
) -> None:
    if metadata is None:
        return

    assert isinstance(
        metadata, list
    ), "Expected `metadata` to be set to None or of type list"
    assert (
        image is not None
    ), "Expected `image` to be passed in if metadata was provided"
    assert (
        aug_image is not None
    ), "Expected `aug_image` to be passed in if metadata was provided"

    transform_bboxes(
        function_name=function_name,
        image=image,
        aug_image=aug_image,
        bboxes_helper_func=bboxes_helper_func,
        **kwargs,
    )

    # Json can't represent tuples, so they're represented as lists, which should
    # be equivalent to tuples. So let's avoid tuples in the metadata by
    # converting any tuples to lists here.
    kwargs_types_fixed = dict(
        (k, list(v)) if isinstance(v, tuple) else (k, v) for k, v in kwargs.items()
    )
    if (
        bboxes_helper_func is not None
        and bboxes_helper_func.__name__ == "spatial_bbox_helper"
    ):
        kwargs_types_fixed.pop("aug_function", None)

    metadata.append(
        {
            "name": function_name,
            "src_width": image.width,
            "src_height": image.height,
            "dst_width": aug_image.width,
            "dst_height": aug_image.height,
            **kwargs_types_fixed,
        }
    )

    intensity_kwargs = {"metadata": metadata[-1], **kwargs}
    metadata[-1]["intensity"] = getattr(
        imintensity, f"{function_name}_intensity", lambda **_: 0.0
    )(**intensity_kwargs)
