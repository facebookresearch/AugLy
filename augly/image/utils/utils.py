#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
import math
import os
from typing import List, Optional, Tuple, Union

import augly.utils as utils
import numpy as np
from PIL import Image

JPEG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG"]


def validate_and_load_image(image: Union[str, Image.Image]) -> Image.Image:
    """
    If image is a str, loads the image as a PIL Image and returns it. Otherwise,
    we assert that image is a PIL Image and then return it.
    """
    if isinstance(image, str):
        local_path = utils.pathmgr.get_local_path(image)
        utils.validate_image_path(local_path)
        return Image.open(local_path)

    assert isinstance(
        image, Image.Image
    ), "Expected type PIL.Image.Image for variable 'image'"

    return image


def ret_and_save_image(image: Image.Image, output_path: Optional[str]) -> Image.Image:
    if output_path is not None:
        if any(output_path.endswith(extension) for extension in JPEG_EXTENSIONS):
            image = image.convert("RGB")

        utils.validate_output_path(output_path)
        image.save(output_path)

    return image


def get_template_and_bbox(
    template_filepath: str, template_bboxes_filepath: str
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    template_key = os.path.basename(template_filepath)
    local_template_path = utils.pathmgr.get_local_path(template_filepath)
    template = Image.open(local_template_path)
    local_bbox_path = utils.pathmgr.get_local_path(template_bboxes_filepath)
    bbox = json.load(open(local_bbox_path, "rb"))[template_key]

    return template, bbox


def rotated_rect_with_max_area(w: int, h: int, angle: float) -> Tuple[float, float]:
    """
    Computes the width and height of the largest possible axis-aligned
    rectangle (maximal area) within the rotated rectangle

    source:
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders # noqa: B950
    """
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a = abs(math.sin(math.radians(angle)))
    cos_a = abs(math.cos(math.radians(angle)))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def pad_with_black(src: Image.Image, w: int, h: int) -> Image.Image:
    """
    Returns the image src with the x dimension padded to width w if it was
    smaller than w (and likewise for the y dimension with height h)
    """
    curr_w, curr_h = src.size
    dx = max(0, (w - curr_w) // 2)
    dy = max(0, (h - curr_h) // 2)
    padded = Image.new("RGB", (w, h))
    padded.paste(src, (dx, dy, curr_w + dx, curr_h + dy))
    return padded


def resize_and_pad_to_given_size(
    src: Image.Image, w: int, h: int, crop: bool
) -> Image.Image:
    """
    Returns the image src resized & padded with black if needed for the screenshot
    transformation (i.e. if the spot for the image in the template is too small or
    too big for the src image). If crop is True, will crop the src image if necessary
    to fit into the template image; otherwise, will resize if necessary
    """
    curr_w, curr_h = src.size
    if crop:
        dx = (curr_w - w) // 2
        dy = (curr_h - h) // 2
        src = src.crop((dx, dy, w + dx, h + dy))
        curr_w, curr_h = src.size
    elif curr_w > w or curr_h > h:
        resize_factor = min(w / curr_w, h / curr_h)
        new_w = int(curr_w * resize_factor)
        new_h = int(curr_h * resize_factor)
        src = src.resize((new_w, new_h), resample=Image.BILINEAR)
        curr_w, curr_h = src.size
    if curr_w < w or curr_h < h:
        src = pad_with_black(src, w, h)
    return src


def scale_template_image(
    src_w: int,
    src_h: int,
    template_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    max_image_size_pixels: Optional[int],
    crop: bool,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Return template_image, and bbox resized to fit the src image. Takes in the
    width & height of the src image plus the bounding box where the src image
    will be inserted into template_image. If the template bounding box is
    bigger than src image in both dimensions, template_image is scaled down
    such that the dimension that was closest to src_image matches, without
    changing the aspect ratio (and bbox is scaled proportionally). Similarly if
    src image is bigger than the bbox in both dimensions, template_image and
    the bbox are scaled up.
    """
    template_w, template_h = template_image.size
    left, upper, right, lower = bbox
    bbox_w, bbox_h = right - left, lower - upper
    # Scale up/down template_image & bbox
    if crop:
        resize_factor = min(src_w / bbox_w, src_h / bbox_h)
    else:
        resize_factor = max(src_w / bbox_w, src_h / bbox_h)

    # If a max image size is provided & the resized template image would be too large,
    # resize the template image to the max image size.
    if max_image_size_pixels is not None:
        template_size = template_w * template_h
        if template_size * resize_factor ** 2 > max_image_size_pixels:
            resize_factor = math.sqrt(max_image_size_pixels / template_size)

    template_w = int(template_w * resize_factor)
    template_h = int(template_h * resize_factor)
    bbox_w, bbox_h = int(bbox_w * resize_factor), int(bbox_h * resize_factor)
    left, upper = int(left * resize_factor), int(upper * resize_factor)
    right, lower = left + bbox_w, upper + bbox_h
    bbox = (left, upper, right, lower)
    template_image = template_image.resize(
        (template_w, template_h), resample=Image.BILINEAR
    )
    return template_image, bbox


def square_center_crop(src: Image.Image) -> Image.Image:
    """Returns a square crop of the center of the image"""
    w, h = src.size
    smallest_edge = min(w, h)
    dx = (w - smallest_edge) // 2
    dy = (h - smallest_edge) // 2
    return src.crop((dx, dy, dx + smallest_edge, dy + smallest_edge))


def compute_transform_coeffs(
    src_coords: List[Tuple[int, int]], dst_coords: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Given the starting & desired corner coordinates, computes the
    coefficients required by the perspective transform.
    """
    matrix = []
    for sc, dc in zip(src_coords, dst_coords):
        matrix.append([dc[0], dc[1], 1, 0, 0, 0, -sc[0] * dc[0], -sc[0] * dc[1]])
        matrix.append([0, 0, 0, dc[0], dc[1], 1, -sc[1] * dc[0], -sc[1] * dc[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(src_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def compute_stripe_mask(
    src_w: int, src_h: int, line_width: float, line_angle: float, line_density: float
) -> np.ndarray:
    """
    Given stripe parameters such as stripe width, angle, and density, returns
    a binary mask of the same size as the source image indicating the location
    of stripes. This implementation is inspired by
    https://stackoverflow.com/questions/34043381/how-to-create-diagonal-stripe-patterns-and-checkerboard-patterns
    """
    line_angle *= math.pi / 180
    line_distance = (1 - line_density) * min(src_w, src_h)

    y_period = math.cos(line_angle) / line_distance
    x_period = math.sin(line_angle) / line_distance
    y_coord_range = np.arange(0, src_h) - src_h / 2
    x_coord_range = np.arange(0, src_w) - src_w / 2
    x_grid_coords, y_grid_coords = np.meshgrid(x_coord_range, y_coord_range)

    if abs(line_angle) == math.pi / 2 or abs(line_angle) == 3 * math.pi / 2:
        # Compute mask for vertical stripes
        softmax_mask = (np.cos(2 * math.pi * x_period * x_grid_coords) + 1) / 2
    elif line_angle == 0 or abs(line_angle) == math.pi:
        # Compute mask for horizontal stripes
        softmax_mask = (np.cos(2 * math.pi * y_period * y_grid_coords) + 1) / 2
    else:
        # Compute mask for diagonal stripes
        softmax_mask = (
            np.cos(2 * math.pi * (x_period * x_grid_coords + y_period * y_grid_coords))
            + 1
        ) / 2

    binary_mask = softmax_mask > (math.cos(math.pi * line_width) + 1) / 2

    return binary_mask
