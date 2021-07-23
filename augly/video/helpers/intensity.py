#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, Optional, Tuple

import augly.image.intensity as imint
import augly.image.utils as imutils
from augly.video.helpers import get_video_info


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


def add_noise_intensity(level: int, **kwargs) -> float:
    assert (
        isinstance(level, (float, int)) and 0 <= level <= 100
    ), "level must be a number in [0, 100]"

    return (level / 100) * 100.0


def apply_lambda_intensity(aug_function: str, **kwargs) -> float:
    intensity_func = globals().get(f"{aug_function}_intensity")
    return intensity_func(**kwargs) if intensity_func else 100.0


def audio_swap_intensity(offset: float, **kwargs) -> float:
    return (1.0 - offset) * 100.0


def blend_videos_intensity(opacity: float, overlay_size: float, **kwargs) -> float:
    return imint.overlay_media_intensity_helper(opacity, overlay_size)


def blur_intensity(sigma: int, **kwargs) -> float:
    assert (
        isinstance(sigma, (float, int)) and sigma >= 0
    ), "sigma must be a non-negative number"

    max_sigma = 100
    return min((sigma / max_sigma) * 100.0, 100.0)


def brightness_intensity(level: float, **kwargs) -> float:
    assert (
        isinstance(level, (float, int)) and -1 <= level <= 1
    ), "level must be a number in [-1, 1]"

    return abs(level) * 100.0


def change_aspect_ratio_intensity(
    ratio: float, metadata: Dict[str, Any], **kwargs
) -> float:
    assert (
        isinstance(ratio, (float, int)) and ratio > 0
    ), "ratio must be a positive number"

    current_ratio = metadata["src_width"] / metadata["src_height"]
    max_ratio_change = 10.0
    ratio_change = abs(ratio - current_ratio)
    return min((ratio_change / max_ratio_change) * 100.0, 100.0)


def change_video_speed_intensity(factor: float, **kwargs):
    assert (
        isinstance(factor, (float, int)) and factor > 0
    ), "factor must be a positive number"

    if factor == 1.0:
        return 0.0
    max_factor = 10.0
    speed_change_factor = factor if factor > 1 else 1 / factor
    return min((speed_change_factor / max_factor) * 100.0, 100.0)


def color_jitter_intensity(
    brightness_factor: float, contrast_factor: float, saturation_factor: float, **kwargs
) -> float:
    assert (
        isinstance(brightness_factor, (float, int)) and -1 <= brightness_factor <= 1
    ), "brightness_factor must be a number in [-1, 1]"
    assert (
        isinstance(contrast_factor, (float, int)) and -1000 <= contrast_factor <= 1000
    ), "contrast_factor must be a number in [-1000, 1000]"
    assert (
        isinstance(saturation_factor, (float, int)) and 0 <= saturation_factor <= 3
    ), "saturation_factor must be a number in [0, 3]"

    brightness_intensity = abs(brightness_factor)
    contrast_intensity = abs(contrast_factor) / 1000
    saturation_intensity = saturation_factor / 3
    return (brightness_intensity * contrast_intensity * saturation_intensity) * 100.0


def concat_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return time_crop_or_pad_intensity_helper(metadata)


def contrast_intensity(level: float, **kwargs) -> float:
    assert (
        isinstance(level, (float, int)) and -1000 <= level <= 1000
    ), "level must be a number in [-1000, 1000]"

    return (abs(level) / 1000) * 100.0


def crop_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def encoding_quality_intensity(quality: int, **kwargs):
    assert (
        isinstance(quality, int) and 0 <= quality <= 51
    ), "quality must be a number in [0, 51]"
    return (quality / 51) * 100.0


def fps_intensity(fps: int, metadata: Dict[str, Any], **kwargs):
    assert isinstance(fps, (float, int)), "fps must be a number"

    src_fps = metadata["src_fps"]
    return min(((src_fps - fps) / src_fps) * 100.0, 100.0)


def grayscale_intensity(**kwargs):
    return 100.0


def hflip_intensity(**kwargs):
    return 100.0


def hstack_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def identity_function_intensity(**kwargs) -> float:
    return 0.0


def insert_in_background_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return time_crop_or_pad_intensity_helper(metadata)


def loop_intensity(num_loops: int, **kwargs) -> float:
    max_num_loops = 100
    return min((num_loops / max_num_loops) * 100.0, 100.0)


def meme_format_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def overlay_intensity(
    overlay_size: Optional[float], overlay_path: str, metadata: Dict[str, Any], **kwargs
) -> float:
    assert (
        overlay_size is None or (
            isinstance(overlay_size, (float, int)) and 0 < overlay_size <= 1
        )
    ), "overlay_size must be a value in the range (0, 1]"
    if overlay_size is not None:
        return (overlay_size ** 2) * 100.0

    try:
        img = imutils.validate_and_load_image(overlay_path)
        overlay_area = img.width * img.height
    except Exception:
        video_info = get_video_info(overlay_path)
        overlay_area = video_info["width"] * video_info["height"]
    src_area = metadata["src_width"] * metadata["src_height"]
    return min((overlay_area / src_area) * 100.0, 100.0)


def overlay_dots_intensity(num_dots: int, **kwargs) -> float:
    max_num_dots = 10000
    return min((num_dots / max_num_dots) * 100.0, 100.0)


def overlay_emoji_intensity(
    emoji_size: float, opacity: float, metadata: Dict[str, Any], **kwargs
) -> float:
    assert (
        isinstance(emoji_size, (float, int)) and 0 <= emoji_size <= 1
    ), "emoji_size must be a number in [0, 1]"
    assert (
        isinstance(opacity, (float, int)) and 0 <= opacity <= 1
    ), "opacity must be a number in [0, 1]"

    video_area = metadata["dst_width"] * metadata["dst_height"]
    emoji_width = min(metadata["dst_width"], metadata["dst_height"] * emoji_size)
    emoji_height = metadata["dst_height"] * emoji_size
    emoji_area = emoji_width * emoji_height
    area_intensity = emoji_area / video_area
    return area_intensity * opacity * 100.0


def overlay_onto_background_video_intensity(
    overlay_size: Optional[float],
    metadata: Dict[str, Any],
    **kwargs,
) -> float:
    if overlay_size is not None:
        return (1 - overlay_size ** 2) * 100.0

    src_area = metadata["src_width"] * metadata["src_height"]
    dst_area = metadata["dst_width"] * metadata["dst_height"]
    return min(100.0, max(0.0, 1.0 - src_area / dst_area) * 100.0)


def overlay_onto_screenshot_intensity(
    template_filepath: str,
    template_bboxes_filepath: str,
    metadata: Dict[str, Any],
    **kwargs,
) -> float:
    _, bbox = imutils.get_template_and_bbox(template_filepath, template_bboxes_filepath)
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    dst_area = metadata["dst_width"] * metadata["dst_height"]
    return ((dst_area - bbox_area) / dst_area) * 100.0


def overlay_shapes_intensity(
    topleft: Optional[Tuple[float, float]],
    bottomright: Optional[Tuple[float, float]],
    num_shapes: int,
    **kwargs,
) -> float:
    return distractor_overlay_intensity_helper(topleft, bottomright, num_shapes)


def overlay_text_intensity(
    topleft: Optional[Tuple[float, float]],
    bottomright: Optional[Tuple[float, float]],
    **kwargs,
) -> float:
    return distractor_overlay_intensity_helper(topleft, bottomright, 1)


def pad_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def perspective_transform_and_shake_intensity(
    sigma: float, shake_radius: float, **kwargs
) -> float:
    assert (
        isinstance(sigma, (float, int)) and sigma >= 0
    ), "sigma must be a non-negative number"
    assert (
        isinstance(shake_radius, (float, int)) and shake_radius >= 0
    ), "shake_radius must be a non-negative number"

    max_sigma_val = 100
    max_shake_radius_val = 100
    sigma_intensity = sigma / max_sigma_val
    shake_radius_intensity = shake_radius / max_shake_radius_val
    return min((sigma_intensity * shake_radius_intensity) * 100.0, 100.0)


def pixelization_intensity(ratio: float, **kwargs) -> float:
    assert (
        isinstance(ratio, (float, int)) and 0 <= ratio <= 1
    ), "ratio must be a number in [0, 1]"
    return (1 - ratio) * 100.0


def remove_audio_intensity(**kwargs) -> float:
    return 100.0

def replace_with_background_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    """
    The intensity of replace_with_background is the fraction of the source video duration
    that was replaced with background. Because the overall duration of the video is preserved,
    the background segments together must be shorter than the source duration so the intensity is never
    greater than 100.
    """
    src_duration = metadata["src_duration"]
    total_bg_duration = metadata["starting_background_duration"] + metadata["ending_background_duration"]
    return min((total_bg_duration / src_duration) * 100.0, 100.0)

def replace_with_color_frames_intensity(
    duration_factor: float, offset_factor: float, **kwargs
) -> float:
    assert (
        isinstance(duration_factor, (float, int)) and 0 <= duration_factor <= 1
    ), "duration_factor must be a number in [0, 1]"
    assert (
        isinstance(offset_factor, (float, int)) and 0 <= offset_factor <= 1
    ), "offset_factor must be a number in [0, 1]"
    # The proportion of the video that is replaced by color frames is generally
    # equal to duration factor, unless offset_factor + duration_factor > 1, in
    # which case it will be 1 - offset_factor.
    return min(duration_factor, 1 - offset_factor) * 100.0


def resize_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def rotate_intensity(degrees: float, **kwargs) -> float:
    assert isinstance(degrees, (float, int)), "degrees must be a number"

    max_degrees_val = 180
    degrees = abs(degrees) % 180
    return (degrees / max_degrees_val) * 100.0


def scale_intensity(factor: float, **kwargs) -> float:
    assert (
        isinstance(factor, (float, int)) and factor > 0
    ), "factor must be a positive number"

    if factor == 1.0:
        return 0.0
    max_factor_val = 10.0
    scale_factor = factor if factor > 1 else 1 / factor
    return min((scale_factor / max_factor_val) * 100.0, 100.0)


def shift_intensity(x_factor: float, y_factor: float, **kwargs) -> float:
    assert (
        isinstance(x_factor, (float, int))
        and 0 <= x_factor <= 1
        and isinstance(y_factor, (float, int))
        and 0 <= y_factor <= 1
    ), "x_factor & y_factor must be positive numbers in [0, 1]"

    return (1 - x_factor) * (1 - y_factor) * 100.0


def time_crop_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return time_crop_or_pad_intensity_helper(metadata)


def time_decimate_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return time_crop_or_pad_intensity_helper(metadata)


def trim_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return time_crop_or_pad_intensity_helper(metadata)


def vflip_intensity(**kwargs) -> float:
    return 100.0


def vstack_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return imint.resize_intensity_helper(metadata)


def distractor_overlay_intensity_helper(
    topleft: Optional[Tuple[float, float]],
    bottomright: Optional[Tuple[float, float]],
    num_overlay_content: int,
    **kwargs,
):
    """
    Computes intensity of any distractor-type transform, which adds some kind
    of media (images, emojis, text, dots, logos) on top of the src video within
    a specified bounding box.
    """
    assert topleft is None or all(
        0.0 <= t <= 1.0 for t in topleft
    ), "Topleft must be in the range [0, 1]"
    assert bottomright is None or all(
        0.0 <= b <= 1.0 for b in bottomright
    ), "Bottomright must be in the range [0, 1]"
    assert (
        isinstance(num_overlay_content, int) and num_overlay_content >= 0
    ), "num_overlay_content must be a nonnegative int"

    if topleft is None or bottomright is None:
        return 100.0

    max_num_overlay_content_val = 100
    num_overlay_content_intensity = num_overlay_content / max_num_overlay_content_val

    x1, y1 = topleft
    x2, y2 = bottomright
    distractor_area = (x2 - x1) * (y2 - y1)
    return min((distractor_area * num_overlay_content_intensity) * 100.0, 100.0)


def time_crop_or_pad_intensity_helper(metadata: Dict[str, Any]) -> float:
    """
    Computes intensity of a transform that consists of temporal cropping or
    padding. For these types of transforms the intensity is defined as the
    percentage of video time that has been cut out (for cropping) or added
    (for padding). When computing the percentage, the denominator should be
    the longer of the src & dst durations so the resulting percentage isn't
    greater than 100.
    """
    dst_duration = metadata["dst_duration"]
    src_duration = metadata["src_duration"]
    larger_duration = max(src_duration, dst_duration)
    return (abs(dst_duration - src_duration) / larger_duration) * 100.0
