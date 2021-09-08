#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import augly.image.intensity as imintensity
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
        assert (
            len(bbox) == 4 and all(isinstance(x, (float, int)) for x in bbox)
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


def meme_format_bboxes_helper(
    bbox: Tuple, src_w: int, src_h: int, caption_height: int, **kwargs
) -> Tuple:
    left_f, upper_f, right_f, lower_f = bbox
    y_off = caption_height / src_h
    new_h = 1.0 + y_off
    return left_f, (upper_f + y_off) / new_h, right_f, (lower_f + y_off) / new_h


def overlay_onto_background_image_bboxes_helper(
    bbox: Tuple, overlay_size: float, x_pos: float, y_pos: float, **kwargs
) -> Tuple:
    left_factor, upper_factor, right_factor, lower_factor = bbox
    return (
        max(0, left_factor * overlay_size + x_pos),
        max(0, upper_factor * overlay_size + y_pos),
        min(1, right_factor * overlay_size + x_pos),
        min(1, lower_factor * overlay_size + y_pos),
    )


def pad_bboxes_helper(bbox: Tuple, w_factor: float, h_factor: float, **kwargs) -> Tuple:
    left_factor, upper_factor, right_factor, lower_factor = bbox
    new_w = 1 + 2 * w_factor
    new_h = 1 + 2 * h_factor
    return (
        (left_factor + w_factor) / new_w,
        (upper_factor + h_factor) / new_h,
        (right_factor + w_factor) / new_w,
        (lower_factor + h_factor) / new_h,
    )


def transform_bbox(
    bbox: Tuple, function_name: str, src_w: int, src_h: int, **kwargs
) -> Tuple:
    left_factor, upper_factor, right_factor, lower_factor = bbox
    if function_name == "hflip":
        return (1 - right_factor, upper_factor, 1 - left_factor, lower_factor)
    elif function_name == "meme_format":
        return meme_format_bboxes_helper(bbox, src_w=src_w, src_h=src_h, **kwargs)
    elif function_name == "overlay_onto_background_image":
        return overlay_onto_background_image_bboxes_helper(bbox, **kwargs)
    elif function_name == "pad":
        return pad_bboxes_helper(bbox, **kwargs)
    elif function_name == "pad_square":
        w_factor, h_factor = 0, 0
        if src_w < src_h:
            w_factor = (src_h - src_w) / (2 * src_w)
        else:
            h_factor = (src_w - src_h) / (2 * src_h)
        return pad_bboxes_helper(bbox, w_factor=w_factor, h_factor=h_factor)
    elif function_name == "vflip":
        return (left_factor, 1 - lower_factor, right_factor, 1 - upper_factor)
    # TODO: add cases for all image transforms that modify the bboxes
    return bbox


def check_for_gone_bboxes(transformed_bboxes: List[Tuple]) -> List[Optional[Tuple]]:
    """
    When a bounding box is cropped out of the image or something is overlaid
    which obfuscates it, we consider the bbox to no longer be visible/valid, so
    we will return it as None
    """
    checked_bboxes = []
    for i in range(len(transformed_bboxes)):
        left_factor, upper_factor, right_factor, lower_factor = transformed_bboxes[i]
        checked_bboxes.append(
            None
            if left_factor >= right_factor or upper_factor >= lower_factor
            else transformed_bboxes[i]
        )
    return checked_bboxes


def transform_bboxes(
    dst_bboxes: Optional[List[Tuple]],
    bbox_format: Optional[str],
    function_name: str,
    image: Image.Image,
    aug_image: Image.Image,
    **kwargs,
) -> None:
    if dst_bboxes is None:
        return

    assert (
        bbox_format is not None
        and bbox_format in ["pascal_voc", "pascal_voc_norm", "coco", "yolo"]
    ), "bbox_format must be specified if bboxes are passed in and must be a supported format"

    src_w, src_h = image.size
    aug_w, aug_h = aug_image.size
    norm_bboxes = validate_and_normalize_bboxes(dst_bboxes, bbox_format, src_w, src_h)
    transformed_norm_bboxes = [
        transform_bbox(bbox, function_name, src_w, src_h, **kwargs)
        for bbox in norm_bboxes
    ]
    transformed_norm_bboxes = check_for_gone_bboxes(transformed_norm_bboxes)
    convert_bboxes(transformed_norm_bboxes, dst_bboxes, bbox_format, aug_w, aug_h)


def get_func_kwargs(
    metadata: Optional[List[Dict[str, Any]]], local_kwargs: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    bboxes = local_kwargs.pop("bboxes")
    func_kwargs = deepcopy(local_kwargs)
    func_kwargs.pop("metadata")
    func_kwargs["src_bboxes"] = deepcopy(bboxes)
    func_kwargs["dst_bboxes"] = bboxes
    func_kwargs.update(**deepcopy(kwargs))
    return func_kwargs


def get_metadata(
    metadata: Optional[List[Dict[str, Any]]],
    function_name: str,
    image: Image.Image,
    aug_image: Image.Image,
    **kwargs,
) -> None:
    transform_bboxes(
        function_name=function_name, image=image, aug_image=aug_image, **kwargs
    )

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

    # Json can't represent tuples, so they're represented as lists, which should
    # be equivalent to tuples. So let's avoid tuples in the metadata by
    # converting any tuples to lists here.
    kwargs_types_fixed = dict(
        (k, list(v)) if isinstance(v, tuple) else (k, v) for k, v in kwargs.items()
    )

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
