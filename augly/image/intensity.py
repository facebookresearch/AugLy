#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import augly.image.utils as imutils
import numpy as np
from PIL import Image


"""
This file contains 'intensity' functions for each of our augmentations.
Intensity functions give a float representation of how intense a particular
transform called with particular parameters is.

Each intensity function expects as parameters the kwargs the corresponding
transform was called with, as well as the metadata dictionary computed by the
transform (e.g. metadata[-1] after passing a list 'metadata' into the transform).

All intensity functions are normalized to be in [0, 100]. This means we are
assuming a range of valid values for each param - e.g. for change_aspect_ratio
we assume we will never change the aspect ratio of the video by more than 10x,
meaning the range of valid values for `ratio` is [0.1, 10.0], which you can see
is assumed in the intensity function below.
"""


def apply_pil_filter_intensity(**kwargs) -> float:
    return 100.0


def apply_lambda_intensity(aug_function: str, **kwargs) -> float:
    intensity_func = globals().get(f"{aug_function}_intensity")
    return intensity_func(**kwargs) if intensity_func else 100.0


def blur_intensity(radius: int, **kwargs) -> float:
    assert (
        isinstance(radius, (float, int)) and radius >= 0
    ), "radius must be a non-negative number"

    max_radius = 100
    return min((radius / max_radius) * 100.0, 100.0)


def brightness_intensity(factor: float, **kwargs) -> float:
    return mult_factor_intensity_helper(factor)


def change_aspect_ratio_intensity(
    ratio: float, metadata: Dict[str, Any], **kwargs
) -> float:
    assert (
        isinstance(ratio, (float, int)) and ratio > 0
    ), "ratio must be a positive number"

    if ratio == metadata["src_width"] / metadata["src_height"]:
        return 0.0
    max_ratio = 10.0
    ratio = ratio if ratio >= 1 else 1 / ratio
    return min((ratio / max_ratio) * 100.0, 100.0)


def clip_image_size_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def color_jitter_intensity(
    brightness_factor: float, contrast_factor: float, saturation_factor: float, **kwargs
) -> float:
    assert (
        isinstance(brightness_factor, (float, int)) and brightness_factor >= 0
    ), "brightness_factor must be a nonnegative number"
    assert (
        isinstance(contrast_factor, (float, int)) and contrast_factor >= 0
    ), "contrast_factor must be a nonnegative number"
    assert (
        isinstance(saturation_factor, (float, int)) and saturation_factor >= 0
    ), "saturation_factor must be a nonnegative number"

    max_total_factor = 30

    brightness_factor = normalize_mult_factor(brightness_factor)
    contrast_factor = normalize_mult_factor(contrast_factor)
    saturation_factor = normalize_mult_factor(saturation_factor)
    total_factor = brightness_factor + contrast_factor + saturation_factor

    return min((total_factor / max_total_factor) * 100.0, 100.0)


def contrast_intensity(factor: float, **kwargs) -> float:
    return mult_factor_intensity_helper(factor)


def convert_color_intensity(**kwargs) -> float:
    return 100.0


def crop_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def encoding_quality_intensity(quality: int, **kwargs):
    assert (
        isinstance(quality, int) and 0 <= quality <= 100
    ), "quality must be a number in [0, 100]"
    return ((100 - quality) / 100) * 100.0


def grayscale_intensity(**kwargs):
    return 100.0


def hflip_intensity(**kwargs):
    return 100.0


def masked_composite_intensity(
    mask: Optional[Union[str, Image.Image]], metadata: Dict[str, Any], **kwargs
) -> float:
    if mask is None:
        mask_intensity = 1.0
    else:
        mask = imutils.validate_and_load_image(mask)
        mask_arr = np.array(mask)
        # There can be 3 dimensions if the mask is RGBA format, in which case
        # we only care about the last channel (alpha) to determine the mask
        mask_values = mask_arr[:, :, -1] if mask_arr.ndim == 3 else mask_arr
        mask_intensity = np.sum(mask_values > 0) / (
            mask_values.shape[0] * mask_values.shape[1]
        )
    if metadata['transform_function'] is None:
        aug_intensity = 0.0
    else:
        aug_intensity_func = globals().get(f"{metadata['transform_function']}_intensity")
        aug_intensity = (
            aug_intensity_func(**kwargs) / 100.0 if aug_intensity_func is not None else 1.0
        )
    return (aug_intensity * mask_intensity) * 100.0


def meme_format_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def opacity_intensity(level: float, **kwargs) -> float:
    assert (
        isinstance(level, (float, int)) and 0 <= level <= 1
    ), "level must be a number in [0, 1]"

    return (1 - level) * 100.0


def overlay_emoji_intensity(
    emoji_size: float, opacity: float, **kwargs
) -> float:
    return overlay_media_intensity_helper(opacity, emoji_size)


def overlay_image_intensity(
    opacity: float, overlay_size: float, **kwargs
) -> float:
    return overlay_media_intensity_helper(opacity, overlay_size)


def overlay_onto_background_image_intensity(
    opacity: float, overlay_size: float, **kwargs
) -> float:
    return 100.0 - overlay_media_intensity_helper(opacity, overlay_size)


def overlay_onto_screenshot_intensity(
    template_filepath: str,
    template_bboxes_filepath: str,
    metadata: Dict[str, Any],
    **kwargs,
) -> float:
    _, bbox = imutils.get_template_and_bbox(
        template_filepath, template_bboxes_filepath
    )
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    dst_area = metadata["dst_width"] * metadata["dst_height"]
    return min(((dst_area - bbox_area) / dst_area) * 100.0, 100.0)


def overlay_stripes_intensity(
    line_width: float,
    line_angle: float,
    line_density: float,
    line_type: str,
    line_opacity: float,
    metadata: Dict[str, Any],
    **kwargs) -> float:
    binary_mask = imutils.compute_stripe_mask(
        src_w=metadata["src_width"],
        src_h=metadata["src_height"],
        line_width=line_width,
        line_angle=line_angle,
        line_density=line_density,
    )

    if line_type == "dotted":
        # To create dotted effect, multiply mask by stripes in perpendicular direction
        perpendicular_mask = imutils.compute_stripe_mask(
            src_w=metadata["src_width"],
            src_h=metadata["src_height"],
            line_width=line_width,
            line_angle=line_angle + 90,
            line_density=line_density,
        )
        binary_mask *= perpendicular_mask
    elif line_type == "dashed":
        # To create dashed effect, multiply mask by stripes with a larger line
        # width in perpendicular direction
        perpendicular_mask = imutils.compute_stripe_mask(
            src_w=metadata["src_width"],
            src_h=metadata["src_height"],
            line_width=0.7,
            line_angle=line_angle + 90,
            line_density=line_density,
        )
        binary_mask *= perpendicular_mask

    perc_stripes = np.mean(binary_mask)
    return overlay_media_intensity_helper(line_opacity, perc_stripes)


def overlay_text_intensity(opacity: float, font_size: float, **kwargs) -> float:
    return overlay_media_intensity_helper(opacity, font_size)


def pad_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def pad_square_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def perspective_transform_intensity(sigma: float, **kwargs) -> float:
    assert (
        isinstance(sigma, (float, int)) and sigma >= 0
    ), "sigma must be a non-negative number"

    max_sigma_val = 100
    sigma_intensity = sigma / max_sigma_val
    return min(sigma_intensity * 100.0, 100.0)


def pixelization_intensity(ratio: float, **kwargs) -> float:
    assert (
        isinstance(ratio, (float, int)) and ratio > 0
    ), "ratio must be a positive number"
    return min((1 - ratio) * 100.0, 100.0)


def random_noise_intensity(mean: float, var: float, **kwargs) -> float:
    assert isinstance(mean, (float, int)), "mean must be a number"
    assert (
        isinstance(var, (float, int)) and var >= 0
    ), "var must be a non-negative number"

    max_mean_val = 100
    max_var_val = 10
    # Even if mean or var is 0, we want the intensity to be non-zero if the
    # other one is non-zero, so we add a little jitter away from 0
    mean_intensity = max(abs(mean / max_mean_val), 0.01)
    var_intensity = max(var / max_var_val, 0.01)
    return (mean_intensity * var_intensity) * 100.0


def resize_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return resize_intensity_helper(metadata)


def rotate_intensity(degrees: float, **kwargs) -> float:
    assert isinstance(degrees, (float, int)), "degrees must be a number"

    max_degrees_val = 180
    degrees = abs(degrees) % 180
    return (degrees / max_degrees_val) * 100.0


def saturation_intensity(factor: float, **kwargs) -> float:
    return mult_factor_intensity_helper(factor)


def scale_intensity(factor: float, **kwargs) -> float:
    assert (
        isinstance(factor, (float, int)) and factor > 0
    ), "factor must be a positive number"

    if factor == 1.0:
        return 0.0
    max_factor_val = 10.0
    scale_factor = factor if factor > 1 else 1 / factor
    return min((scale_factor / max_factor_val) * 100.0, 100.0)


def sharpen_intensity(factor: float, **kwargs) -> float:
    return mult_factor_intensity_helper(factor)


def shuffle_pixels_intensity(factor: float, **kwargs) -> float:
    return factor * 100.0


def vflip_intensity(**kwargs) -> float:
    return 100.0


def normalize_mult_factor(factor: float) -> float:
    assert (
        isinstance(factor, (float, int)) and factor >= 0
    ), "factor must be a non-negative number"

    if factor == 1:
        return 0.0

    return factor if factor >= 1 else 1 / factor


def mult_factor_intensity_helper(factor: float) -> float:
    factor = normalize_mult_factor(factor)
    max_factor = 10
    return min((factor / max_factor) * 100.0, 100.0)


def overlay_media_intensity_helper(
    opacity: float, overlay_content_size: float
) -> float:
    assert (
        isinstance(opacity, (float, int)) and 0 <= opacity <= 1
    ), "opacity must be a number in [0, 1]"
    assert (
        isinstance(overlay_content_size, (float, int))
        and 0 <= overlay_content_size <= 1
    ), "content size factor must be a number in [0, 1]"

    return (opacity * (overlay_content_size ** 2)) * 100.0


def resize_intensity_helper(metadata: Dict[str, Any]) -> float:
    """
    Computes intensity of any transform that resizes the src image. For these
    types of transforms the intensity is defined as the percentage of image
    area that has been cut out (if cropped/resized to smaller) or added (if
    padding/resized to bigger). When computing the percentage, the denominator
    should be the larger of the src & dst areas so the resulting percentage
    isn't greater than 100.
    """
    src_area = metadata["src_width"] * metadata["src_height"]
    dst_area = metadata["dst_width"] * metadata["dst_height"]
    larger_area = max(src_area, dst_area)
    return (abs(dst_area - src_area) / larger_area) * 100.0
