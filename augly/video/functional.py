#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import math
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from augly import audio as audaugs, image as imaugs, utils
from augly.audio import utils as audutils
from augly.video import helpers, utils as vdutils
from augly.video.augmenters import cv2 as ac, ffmpeg as af


def add_noise(
    video_path: str,
    output_path: Optional[str] = None,
    level: int = 25,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Adds noise to a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param level: noise strength for specific pixel component. Default value is
        25. Allowed range is [0, 100], where 0 indicates no change

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    noise_aug = af.VideoAugmenterByNoise(level)
    noise_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="add_noise", **func_kwargs
        )

    return output_path or video_path


def apply_lambda(
    video_path: str,
    output_path: Optional[str] = None,
    aug_function: Callable[..., Any] = helpers.identity_function,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """
    Apply a user-defined lambda on a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param aug_function: the augmentation function to be applied onto the video
        (should expect a video path and output path as input and output the augmented
        video to the output path. Nothing needs to be returned)

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param **kwargs: the input attributes to be passed into `aug_function`

    @returns: the path to the augmented video
    """
    assert callable(aug_function), (
        repr(type(aug_function).__name__) + " object is not callable"
    )

    func_kwargs = helpers.get_func_kwargs(
        metadata, locals(), video_path, aug_function=aug_function.__name__
    )

    aug_function(video_path, output_path or video_path, **kwargs)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="apply_lambda", **func_kwargs
        )

    return output_path or video_path


def audio_swap(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    offset: float = 0.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Swaps the video audio for the audio passed in provided an offset

    @param video_path: the path to the video to be augmented

    @param audio_path: the iopath uri to the audio you'd like to swap with the
        video's audio

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param offset: starting point in seconds such that an audio clip of offset to
        offset + video_duration is used in the audio swap. Default value is zero

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    audio_swap_aug = af.VideoAugmenterByAudioSwap(audio_path, offset)
    audio_swap_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="audio_swap", **func_kwargs
        )

    return output_path or video_path


def augment_audio(
    video_path: str,
    output_path: Optional[str] = None,
    audio_aug_function: Callable[..., Tuple[np.ndarray, int]] = audaugs.apply_lambda,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **audio_aug_kwargs,
) -> str:
    """
    Augments the audio track of the input video using a given AugLy audio augmentation

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param audio_aug_function: the augmentation function to be applied onto the video's
        audio track. Should have the standard API of an AugLy audio augmentation, i.e.
        expect input audio as a numpy array or path & output path as input, and output
        the augmented audio to the output path

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param audio_aug_kwargs: the input attributes to be passed into `audio_aug`

    @returns: the path to the augmented video
    """
    assert callable(audio_aug_function), (
        repr(type(audio_aug_function).__name__) + " object is not callable"
    )

    func_kwargs = helpers.get_func_kwargs(
        metadata, locals(), video_path, audio_aug_function=audio_aug_function
    )

    if audio_aug_function is not None:
        try:
            func_kwargs["audio_aug_function"] = audio_aug_function.__name__
        except AttributeError:
            func_kwargs["audio_aug_function"] = type(audio_aug_function).__name__

    audio_metadata = []
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        helpers.extract_audio_to_file(video_path, tmpfile.name)
        audio, sr = audutils.validate_and_load_audio(tmpfile.name)
        aug_audio, aug_sr = audio_aug_function(
            audio, sample_rate=sr, metadata=audio_metadata, **audio_aug_kwargs
        )
        audutils.ret_and_save_audio(aug_audio, tmpfile.name, aug_sr)
        audio_swap(video_path, tmpfile.name, output_path=output_path or video_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            audio_metadata=audio_metadata,
            function_name="augment_audio",
            **func_kwargs,
        )

    return output_path or video_path


def blend_videos(
    video_path: str,
    overlay_path: str,
    output_path: Optional[str] = None,
    opacity: float = 0.5,
    overlay_size: float = 1.0,
    x_pos: float = 0.0,
    y_pos: float = 0.0,
    use_second_audio: bool = True,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Overlays a video onto another video at position (width * x_pos, height * y_pos)
    at a lower opacity

    @param video_path: the path to the video to be augmented

    @param overlay_path: the path to the video that will be overlaid onto the
        background video

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param opacity: the lower the opacity, the more transparent the overlaid video

    @param overlay_size: size of the overlaid video is overlay_size * height of
        the background video

    @param x_pos: position of overlaid video relative to the background video width

    @param y_pos: position of overlaid video relative to the background video height

    @param use_second_audio: use the audio of the overlaid video rather than the audio
        of the background video

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    blend_func = functools.partial(
        imaugs.overlay_image,
        opacity=opacity,
        overlay_size=overlay_size,
        x_pos=x_pos,
        y_pos=y_pos,
    )

    vdutils.apply_to_frames(
        blend_func, video_path, overlay_path, output_path, use_second_audio
    )

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="blend_videos", **func_kwargs
        )

    return output_path or video_path


def blur(
    video_path: str,
    output_path: Optional[str] = None,
    sigma: float = 1,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Blurs a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param sigma: horizontal sigma, standard deviation of Gaussian blur

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    blur_aug = af.VideoAugmenterByBlur(sigma)
    blur_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="blur", **func_kwargs)

    return output_path or video_path


def brightness(
    video_path: str,
    output_path: Optional[str] = None,
    level: float = 0.15,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Brightens or darkens a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param level: the value must be a float value in range -1.0 to 1.0, where a
        negative value darkens and positive brightens

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    brightness_aug = af.VideoAugmenterByBrightness(level)
    brightness_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="brightness", **func_kwargs
        )

    return output_path or video_path


def change_aspect_ratio(
    video_path: str,
    output_path: Optional[str] = None,
    ratio: Union[float, str] = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Changes the sample aspect ratio attribute of the video, and resizes the
    video to reflect the new aspect ratio

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param ratio: aspect ratio of the new video, either as a float i.e. width/height,
        or as a string representing the ratio in the form "num:denom"

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    aspect_ratio_aug = af.VideoAugmenterByAspectRatio(ratio)
    aspect_ratio_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="change_aspect_ratio", **func_kwargs
        )

    return output_path or video_path


def change_video_speed(
    video_path: str,
    output_path: Optional[str] = None,
    factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Changes the speed of the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param factor: the factor by which to alter the speed of the video. A factor
        less than one will slow down the video, a factor equal to one won't alter
        the video, and a factor greater than one will speed up the video

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    speed_aug = af.VideoAugmenterBySpeed(factor)
    speed_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="change_video_speed", **func_kwargs
        )

    return output_path or video_path


def color_jitter(
    video_path: str,
    output_path: Optional[str] = None,
    brightness_factor: float = 0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Color jitters the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param brightness_factor: set the brightness expression. The value must be a
        float value in range -1.0 to 1.0. The default value is 0

    @param contrast_factor: set the contrast expression. The value must be a float
        value in range -1000.0 to 1000.0. The default value is 1

    @param saturation_factor: set the saturation expression. The value must be a float
        in range 0.0 to 3.0. The default value is 1

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    color_jitter_aug = af.VideoAugmenterByColorJitter(
        brightness_level=brightness_factor,
        contrast_level=contrast_factor,
        saturation_level=saturation_factor,
    )
    color_jitter_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="color_jitter", **func_kwargs
        )

    return output_path or video_path


def concat(
    video_paths: List[str],
    output_path: Optional[str] = None,
    src_video_path_index: int = 0,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Concatenates videos together. Resizes all other videos to the size of the
    `source` video (video_paths[src_video_path_index]), and modifies the sample
    aspect ratios to match (ffmpeg will fail to concat if SARs don't match)

    @param video_paths: a list of paths to all the videos to be concatenated (in order)

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param src_video_path_index: for metadata purposes, this indicates which video in
        the list `video_paths` should be considered the `source` or original video

    @param transition: optional transition configuration to apply between the clips

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(
        metadata, locals(), video_paths[src_video_path_index]
    )

    concat_aug = af.VideoAugmenterByConcat(
        video_paths,
        src_video_path_index,
        transition,
    )
    concat_aug.add_augmenter(video_paths[src_video_path_index], output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="concat",
            video_path=video_paths[src_video_path_index],
            **func_kwargs,
        )

    return output_path or video_paths[src_video_path_index]


def contrast(
    video_path: str,
    output_path: Optional[str] = None,
    level: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Alters the contrast of a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param level: the value must be a float value in range -1000.0 to 1000.0,
        where a negative value removes contrast and a positive value adds contrast

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    contrast_aug = af.VideoAugmenterByContrast(level)
    contrast_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="contrast", **func_kwargs)

    return output_path or video_path


def crop(
    video_path: str,
    output_path: Optional[str] = None,
    left: float = 0.25,
    top: float = 0.25,
    right: float = 0.75,
    bottom: float = 0.75,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Crops the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param left: left positioning of the crop; between 0 and 1, relative to
        the video width

    @param top: top positioning of the crop; between 0 and 1, relative to
        the video height

    @param right: right positioning of the crop; between 0 and 1, relative to
        the video width

    @param bottom: bottom positioning of the crop; between 0 and 1, relative to
        the video height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    crop_aug = af.VideoAugmenterByCrop(left, top, right, bottom)
    crop_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="crop", **func_kwargs)

    return output_path or video_path


def encoding_quality(
    video_path: str,
    output_path: Optional[str] = None,
    quality: int = 23,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Alters the encoding quality of a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param quality: CRF scale is 0–51, where 0 is lossless, 23 is the default,
        and 51 is worst quality possible. A lower value generally leads to higher
        quality, and a subjectively sane range is 17–28

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    encoding_aug = af.VideoAugmenterByQuality(quality)
    encoding_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="encoding_quality", **func_kwargs
        )

    return output_path or video_path


def fps(
    video_path: str,
    output_path: Optional[str] = None,
    fps: int = 15,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Alters the FPS of a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param fps: the desired output frame rate. Note that a FPS value greater than
        the original FPS of the video will result in an unaltered video

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    fps_aug = af.VideoAugmenterByFPSChange(fps)
    fps_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="fps", **func_kwargs)

    return output_path or video_path


def grayscale(
    video_path: str,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Changes a video to be grayscale

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    grayscale_aug = af.VideoAugmenterByGrayscale()
    grayscale_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="grayscale", **func_kwargs
        )

    return output_path or video_path


def hflip(
    video_path: str,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Horizontally flips a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    hflip_aug = af.VideoAugmenterByHFlip()
    hflip_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="hflip", **func_kwargs)

    return output_path or video_path


def hstack(
    video_path: str,
    second_video_path: str,
    output_path: Optional[str] = None,
    use_second_audio: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Horizontally stacks two videos

    @param video_path: the path to the video that will be stacked to the left

    @param second_video_path: the path to the video that will be stacked to the right

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param use_second_audio: if set to True, the audio of the right video will be
        used instead of the left's

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    hstack_aug = af.VideoAugmenterByStack(second_video_path, use_second_audio, "hstack")
    hstack_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="hstack", **func_kwargs)

    return output_path or video_path


def insert_in_background(
    video_path: str,
    output_path: Optional[str] = None,
    background_path: Optional[str] = None,
    offset_factor: float = 0.0,
    source_percentage: Optional[float] = None,
    seed: Optional[int] = None,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Puts the video in the middle of the background video
    (at offset_factor * background.duration)

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param background_path: the path to the video in which to insert the main
        video. If set to None, the main video will play in the middle of a silent
        video with black frames

    @param offset_factor: the point in the background video in which the main video
        starts to play (this factor is multiplied by the background video duration to
        determine the start point)

    @param source_percentage: when set, source_percentage of the duration
        of the final video (background + source) will be taken up by the
        source video. Randomly crops the background video to the correct duration.
        If the background video isn't long enough to get the desired source_percentage,
        it will be looped.

    @param seed: if provided, this will set the random seed to ensure consistency
        between runs

    @param transition: optional transition configuration to apply between the clips

    @param metadata: if set to be a list, metadata about the function execution including
        its name, the source & dest duration, fps, etc. will be appended to the inputted
        list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    assert (
        0.0 <= offset_factor <= 1.0
    ), "Offset factor must be a value in the range [0.0, 1.0]"

    if source_percentage is not None:
        assert (
            0.0 <= source_percentage <= 1.0
        ), "Source percentage must be a value in the range [0.0, 1.0]"

    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    local_path = utils.pathmgr.get_local_path(video_path)
    utils.validate_video_path(local_path)

    video_info = helpers.get_video_info(local_path)
    video_duration = float(video_info["duration"])
    width, height = video_info["width"], video_info["height"]

    rng = np.random.RandomState(seed) if seed is not None else np.random

    video_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video_path = os.path.join(tmpdir, "in.mp4")
        resized_bg_path = os.path.join(tmpdir, "bg.mp4")

        helpers.add_silent_audio(video_path, tmp_video_path)

        if background_path is None:
            helpers.create_color_video(resized_bg_path, video_duration, height, width)
        else:
            resize(background_path, resized_bg_path, height, width)

        bg_video_info = helpers.get_video_info(resized_bg_path)
        bg_video_duration = float(bg_video_info["duration"])

        bg_start = 0
        bg_end = bg_video_duration
        desired_bg_duration = bg_video_duration
        if source_percentage is not None:
            # desired relationship: percent * (bg_len + s_len) = s_len
            # solve for bg_len -> bg_len = s_len / percent - s_len
            desired_bg_duration = video_duration / source_percentage - video_duration

            # if background vid isn't long enough, loop
            num_loops_needed = math.ceil(desired_bg_duration / bg_video_duration)
            if num_loops_needed > 1:
                loop(resized_bg_path, num_loops=num_loops_needed)
                bg_video_duration *= num_loops_needed

            bg_start = rng.uniform(0, bg_video_duration - desired_bg_duration)
            bg_end = bg_start + desired_bg_duration

        offset = desired_bg_duration * offset_factor

        transition_before = False
        if offset > 0:
            before_path = os.path.join(tmpdir, "before.mp4")
            trim(resized_bg_path, before_path, start=bg_start, end=bg_start + offset)
            video_paths.append(before_path)
            src_video_path_index = 1
            transition_before = True
        else:
            src_video_path_index = 0

        video_paths.append(tmp_video_path)

        transition_after = False
        if bg_start + offset < bg_end:
            after_path = os.path.join(tmpdir, "after.mp4")
            trim(resized_bg_path, after_path, start=bg_start + offset, end=bg_end)
            video_paths.append(after_path)
            transition_after = True

        concat(
            video_paths,
            output_path or video_path,
            src_video_path_index,
            transition=transition,
        )

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="insert_in_background",
            background_video_duration=desired_bg_duration,
            transition_before=transition_before,
            transition_after=transition_after,
            **func_kwargs,
        )

    return output_path or video_path


def insert_in_background_multiple(
    video_path: str,
    output_path: str,
    background_path: str,
    src_ids: List[str],
    additional_video_paths: List[str],
    seed: Optional[int] = None,
    min_source_segment_duration: float = 5.0,
    max_source_segment_duration: float = 20.0,
    min_background_segment_duration: float = 2.0,
    min_result_video_duration: float = 30.0,
    max_result_video_duration: float = 60.0,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Places the video (and the additional videos) in the middle of the background video.

    @param video_path: the path of the main video to be augmented.

    @param output_path: the path in which the output video will be stored.

    @param background_path: the path of the video in which to insert the main
        (and additional) video.

    @param src_ids: the list of identifiers for the main video and additional videos.

    @param additional_video_paths: list of additional video paths to be
        inserted alongside the main video; one clip from each of the input
        videos will be inserted in order.

    @param seed: if provided, this will set the random seed to ensure consistency
        between runs.

    @param min_source_segment_duration: minimum duration in seconds of the source
        segments that will be inserted in the background video.

    @param max_source_segment_duration: maximum duration in seconds of the source
        segments that will be inserted in the background video.

    @param min_background_segment_duration: minimum duration in seconds of a background segment.

    @param min_result_video_duration: minimum duration in seconds of the output video.

    @param max_result_video_duration: maximum duration in seconds of the output video.

    @param transition: optional transition configuration to apply between the clips.

    @param metadata: if set to be a list, metadata about the function execution including
        its name, the source & dest duration, fps, etc. will be appended to the inputted
        list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    if additional_video_paths:
        assert len(additional_video_paths) + 1 == len(
            src_ids
        ), "src_ids need to be specified for the main video and all additional videos."
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    local_path = utils.pathmgr.get_local_path(video_path)
    additional_local_paths = (
        [utils.pathmgr.get_local_path(p) for p in additional_video_paths]
        if additional_video_paths
        else []
    )
    bkg_local_path = utils.pathmgr.get_local_path(background_path)
    src_paths = [
        local_path,
    ] + additional_local_paths

    src_video_durations = np.array(
        [float(helpers.get_video_info(v)["duration"]) for v in src_paths]
    )
    bkg_duration = float(helpers.get_video_info(bkg_local_path)["duration"])

    src_segment_durations = (
        rng.random_sample(len(src_video_durations))
        * (max_source_segment_duration - min_source_segment_duration)
        + min_source_segment_duration
    )
    src_segment_durations = np.minimum(src_segment_durations, src_video_durations)
    src_segment_starts = rng.random(len(src_video_durations)) * (
        src_video_durations - src_segment_durations
    )
    src_segment_ends = src_segment_starts + src_segment_durations

    sum_src_duration = np.sum(src_segment_durations)
    required_result_duration = (
        len(src_segment_durations) + 1
    ) * min_background_segment_duration + sum_src_duration
    if required_result_duration > max_result_video_duration:
        raise ValueError(
            "Failed to generate config for source segments in insert_in_background_multiple."
        )

    duration_budget = max_result_video_duration - required_result_duration
    bkg_budget = rng.random() * duration_budget
    overall_bkg_needed_duration = (
        len(src_segment_durations) + 1
    ) * min_background_segment_duration + bkg_budget

    num_loops_needed = 0
    if overall_bkg_needed_duration > bkg_duration:
        num_loops_needed = math.ceil(overall_bkg_needed_duration / bkg_duration)

    # Now sample insertion points by picking len(src_segment_durations) points in the interval [0, bkg_budget)
    # Then sort the segments and add spacing for the minimum background segment duration.
    bkg_insertion_points = (
        np.sort(rng.random(len(src_segment_durations)) * bkg_budget)
        + np.arange(len(src_segment_durations)) * min_background_segment_duration
    )
    last_bkg_point = overall_bkg_needed_duration
    dst_starts = bkg_insertion_points + np.concatenate(
        (
            [
                0.0,
            ],
            np.cumsum(src_segment_durations)[:-1],
        )
    )

    # Start applying transforms.
    with tempfile.TemporaryDirectory() as tmpdir:
        # First, loop through background video if needed.
        if num_loops_needed > 0:
            buf = os.path.join(tmpdir, "bkg_loop.mp4")
            loop(bkg_local_path, buf, num_loops=num_loops_needed)
            bkg_path = buf
        else:
            bkg_path = bkg_local_path

        bkg_videos = []
        # Sample background segments.
        prev = 0.0
        for i, pt in enumerate(bkg_insertion_points):
            out_path = os.path.join(tmpdir, f"bkg_{i}.mp4")
            trim(bkg_path, out_path, start=prev, end=pt)
            prev = pt
            bkg_videos.append(out_path)

        # last background segment
        last_bkg_path = os.path.join(tmpdir, "bkg_last.mp4")
        trim(bkg_path, last_bkg_path, start=prev, end=last_bkg_point)

        src_videos = []
        # Sample source segments.
        for i, seg in enumerate(zip(src_segment_starts, src_segment_ends)):
            out_path = os.path.join(tmpdir, f"src_{i}.mp4")
            trim(src_paths[i], out_path, start=seg[0], end=seg[1])
            src_videos.append(out_path)

        all_videos = [v for pair in zip(bkg_videos, src_videos) for v in pair] + [
            last_bkg_path,
        ]
        concat(all_videos, output_path, 1, transition=transition)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="insert_in_background_multiple",
            src_segment_starts=src_segment_starts,
            src_segment_ends=src_segment_ends,
            bkg_insertion_points=bkg_insertion_points,
            **func_kwargs,
        )

    return output_path


def replace_with_background(
    video_path: str,
    output_path: Optional[str] = None,
    background_path: Optional[str] = None,
    source_offset: float = 0.0,
    background_offset: float = 0.0,
    source_percentage: float = 0.5,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Replaces the beginning and end of the source video with the background video, keeping the
    total duration of the output video equal to the original duration of the source video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored. If not
        passed in, the original video file will be overwritten

    @param background_path: the path to the video in which to insert the main video.
        If set to None, the main video will play in the middle of a silent video with
        black frames

    @param source_offset: the starting point where the background video transitions to
        the source video. Prior to this point, the source video is replaced with the
        background video. A value of 0 means all background is at the beginning. A value
        of 1 means all background is at the end of the video

    @param background_offset: the starting point from which the background video starts
        to play, as a proportion of the background video duration (i.e. this factor is
        multiplied by the background video duration to determine the start point)

    @param source_percentage: the percentage of the source video that remains unreplaced
        by the background video. The source percentage plus source offset should be less
        than 1. If it is greater, the output video duration will be longer than the source.
        If the background video is not long enough to get the desired source percentage,
        it will be looped

    @param transition: optional transition configuration to apply between the clips

    @param metadata: if set to be a list, metadata about the function execution including
        its name, the source & dest duration, fps, etc. will be appended to the inputted
        list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    assert (
        0.0 <= source_offset <= 1.0
    ), "Source offset factor must be a value in the range [0.0, 1.0]"

    assert (
        0.0 <= background_offset <= 1.0
    ), "Background offset factor must be a value in the range [0.0, 1.0]"

    assert (
        0.0 <= source_percentage <= 1.0
    ), "Source percentage must be a value in the range [0.0, 1.0]"

    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    local_path = utils.pathmgr.get_local_path(video_path)
    utils.validate_video_path(local_path)

    video_info = helpers.get_video_info(video_path)
    video_duration = float(video_info["duration"])
    width, height = video_info["width"], video_info["height"]

    video_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video_path = os.path.join(tmpdir, "in.mp4")
        resized_bg_path = os.path.join(tmpdir, "bg.mp4")

        # create bg video
        if background_path is None:
            helpers.create_color_video(resized_bg_path, video_duration, height, width)
        else:
            resize(background_path, resized_bg_path, height, width)
            helpers.add_silent_audio(resized_bg_path)

        bg_video_info = helpers.get_video_info(resized_bg_path)
        bg_video_duration = float(bg_video_info["duration"])
        src_video_path_index = 1

        final_bg_len = video_duration * (1 - source_percentage)
        # if desired bg video too short, loop bg video
        num_loops_needed = math.ceil(final_bg_len / bg_video_duration)
        if num_loops_needed > 1:
            loop(resized_bg_path, num_loops=num_loops_needed)

        first_bg_segment_len = source_offset * final_bg_len
        last_bg_segment_len = final_bg_len - first_bg_segment_len
        # calculate bg start and end times of bg in output video
        bg_start = background_offset * bg_video_duration
        src_start = first_bg_segment_len
        src_length = source_percentage * video_duration
        src_end = src_start + src_length

        # add pre src background segment
        if source_offset > 0:
            before_path = os.path.join(tmpdir, "before.mp4")
            trim(
                resized_bg_path,
                before_path,
                start=bg_start,
                end=bg_start + first_bg_segment_len,
            )
            video_paths.append(before_path)
            src_video_path_index = 1
        else:
            src_video_path_index = 0

        # trim source to length satisfying source_percentage
        helpers.add_silent_audio(video_path, tmp_video_path)
        trimmed_src_path = os.path.join(tmpdir, "trim_src.mp4")
        trim(tmp_video_path, trimmed_src_path, start=src_start, end=src_end)
        video_paths.append(trimmed_src_path)

        # add post src background segment
        if source_offset < 1:
            after_path = os.path.join(tmpdir, "after.mp4")
            trim(
                resized_bg_path,
                after_path,
                start=bg_start + src_start,
                end=bg_start + src_start + last_bg_segment_len,
            )
            video_paths.append(after_path)

        concat(
            video_paths,
            output_path or video_path,
            src_video_path_index,
            transition=transition,
        )

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="replace_with_background",
            starting_background_duration=first_bg_segment_len,
            source_duration=src_length,
            ending_background_duration=last_bg_segment_len,
            **func_kwargs,
        )

    return output_path or video_path


def loop(
    video_path: str,
    output_path: Optional[str] = None,
    num_loops: int = 0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Loops a video `num_loops` times

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param num_loops: the number of times to loop the video. 0 means that the video
        will play once (i.e. no loops)

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    loop_aug = af.VideoAugmenterByLoops(num_loops)
    loop_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="loop", **func_kwargs)

    return output_path or video_path


def meme_format(
    video_path: str,
    output_path: Optional[str] = None,
    text: str = "LOL",
    font_file: str = utils.MEME_DEFAULT_FONT,
    opacity: float = 1.0,
    text_color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    caption_height: int = 250,
    meme_bg_color: Tuple[int, int, int] = utils.WHITE_RGB_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Creates a new video that looks like a meme, given text and video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param text: the text to be overlaid/used in the meme. note: if using a very
        long string, please add in newline characters such that the text remains
        in a readable font size

    @param font_file: iopath uri to the .ttf font file

    @param opacity: the lower the opacity, the more transparent the text

    @param text_color: color of the text in RGB values

    @param caption_height: the height of the meme caption

    @param meme_bg_color: background color of the meme caption in RGB values

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    meme_func = functools.partial(
        imaugs.meme_format,
        text=text,
        font_file=font_file,
        opacity=opacity,
        text_color=text_color,
        caption_height=caption_height,
        meme_bg_color=meme_bg_color,
    )

    vdutils.apply_to_each_frame(meme_func, video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="meme_format", **func_kwargs
        )

    return output_path or video_path


def overlay(
    video_path: str,
    overlay_path: str,
    output_path: Optional[str] = None,
    overlay_size: Optional[float] = None,
    x_factor: float = 0.0,
    y_factor: float = 0.0,
    use_overlay_audio: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Overlays media onto the video at position (width * x_factor, height * y_factor)

    @param video_path: the path to the video to be augmented

    @param overlay_path: the path to the media (image or video) that will be
        overlaid onto the video

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param overlay_size: size of the overlaid media with respect to the background
        video. If set to None, the original size of the overlaid media is used

    @param x_factor: specifies where the left side of the overlaid media should be
        placed, relative to the video width

    @param y_factor: specifies where the top side of the overlaid media should be
        placed, relative to the video height

    @param use_overlay_audio: if set to True and the media type is a video, the audio
        of the overlaid video will be used instead of the main/background video's audio

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    local_path = utils.pathmgr.get_local_path(video_path)

    overlay_path = utils.pathmgr.get_local_path(overlay_path)
    tmp_overlay_path = None
    if overlay_size is not None:
        assert 0 < overlay_size <= 1, "overlay_size must be a value in the range (0, 1]"

        video_info = helpers.get_video_info(local_path)
        overlay_h = int(video_info["height"] * overlay_size)
        overlay_w = int(video_info["width"] * overlay_size)

        _, tmp_overlay_path = tempfile.mkstemp(suffix=os.path.splitext(overlay_path)[1])

        if utils.is_image_file(overlay_path):
            imaugs.resize(overlay_path, tmp_overlay_path, overlay_w, overlay_h)
        else:
            resize(overlay_path, tmp_overlay_path, overlay_h, overlay_w)

    overlay_aug = af.VideoAugmenterByOverlay(
        tmp_overlay_path or overlay_path, x_factor, y_factor, use_overlay_audio
    )
    overlay_aug.add_augmenter(local_path, output_path)

    if tmp_overlay_path:
        os.remove(tmp_overlay_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="overlay", **func_kwargs)

    return output_path or video_path


def overlay_dots(
    video_path: str,
    output_path: Optional[str] = None,
    num_dots: int = 100,
    dot_type: str = "colored",
    random_movement: bool = True,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """
    Overlays dots onto a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param num_dots: the number of dots to add to each frame

    @param dot_type: specify if you would like "blur" or "colored"

    @param random_movement: whether or not you want the dots to randomly move around
        across the frame or to move across in a "linear" way

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    dots_aug = ac.VideoDistractorByDots(num_dots, dot_type, random_movement)
    vdutils.apply_cv2_augmenter(dots_aug, video_path, output_path, **kwargs)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="overlay_dots", **func_kwargs
        )

    return output_path or video_path


def overlay_emoji(
    video_path: str,
    output_path: Optional[str] = None,
    emoji_path: str = utils.EMOJI_PATH,
    x_factor: float = 0.4,
    y_factor: float = 0.4,
    opacity: float = 1.0,
    emoji_size: float = 0.15,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Overlays an emoji onto each frame of a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param emoji_path: iopath uri to the emoji image

    @param x_factor: specifies where the left side of the emoji should be placed,
        relative to the video width

    @param y_factor: specifies where the top side of the emoji should be placed,
        relative to the video height

    @param opacity: opacity of the emoji image

    @param emoji_size: emoji size relative to the height of the video frame

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    local_path = utils.pathmgr.get_local_path(video_path)

    utils.validate_video_path(video_path)
    video_info = helpers.get_video_info(local_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_emoji_path = utils.pathmgr.get_local_path(emoji_path, cache_dir=tmpdir)
        utils.validate_image_path(local_emoji_path)

        emoji_output_path = os.path.join(tmpdir, "modified_emoji.png")

        imaugs.resize(
            local_emoji_path,
            output_path=emoji_output_path,
            height=int(emoji_size * video_info["height"]),
            width=int(emoji_size * video_info["height"]),
        )
        imaugs.opacity(emoji_output_path, output_path=emoji_output_path, level=opacity)

        overlay(
            video_path,
            emoji_output_path,
            output_path,
            x_factor=x_factor,
            y_factor=y_factor,
        )

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="overlay_emoji", **func_kwargs
        )

    return output_path or video_path


def overlay_onto_background_video(
    video_path: str,
    background_path: str,
    output_path: Optional[str] = None,
    overlay_size: Optional[float] = 0.7,
    x_factor: float = 0.0,
    y_factor: float = 0.0,
    use_background_audio: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Overlays the video onto a background video, pointed to by background_path.

    @param video_path: the path to the video to be augmented

    @param background_path: the path to the background video

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param overlay_size: size of the overlaid media with respect to the background
        video. If set to None, the original size of the overlaid media is used

    @param x_factor: specifies where the left side of the overlaid media should be
        placed, relative to the video width

    @param y_factor: specifies where the top side of the overlaid media should be
        placed, relative to the video height

    @param use_background_audio: if set to True and the media type is a video, the
        audio of the background video will be used instead of the src video's audio

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    overlay(
        video_path=background_path,
        overlay_path=video_path,
        output_path=output_path or video_path,
        overlay_size=overlay_size,
        x_factor=x_factor,
        y_factor=y_factor,
        use_overlay_audio=not use_background_audio,
    )

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="overlay_onto_background_video",
            **func_kwargs,
        )

    return output_path or video_path


def overlay_onto_screenshot(
    video_path: str,
    output_path: Optional[str] = None,
    template_filepath: str = utils.TEMPLATE_PATH,
    template_bboxes_filepath: str = utils.BBOXES_PATH,
    max_image_size_pixels: Optional[int] = None,
    crop_src_to_fit: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Overlays the video onto a screenshot template so it looks like it was
    screen-recorded on Instagram

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param template_filepath: iopath uri to the screenshot template

    @param template_bboxes_filepath: iopath uri to the file containing the bounding
        box for each template

    @param max_image_size_pixels: if provided, the template image and/or src video
        will be scaled down to avoid an output image with an area greater than this
        size (in pixels)

    @param crop_src_to_fit: if True, the src image will be cropped if necessary to
        fit into the template image. If False, the src image will instead be resized
        if necessary

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    sc_func = functools.partial(
        imaugs.overlay_onto_screenshot,
        template_filepath=template_filepath,
        template_bboxes_filepath=template_bboxes_filepath,
        max_image_size_pixels=max_image_size_pixels,
        crop_src_to_fit=crop_src_to_fit,
    )

    vdutils.apply_to_each_frame(sc_func, video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="overlay_onto_screenshot", **func_kwargs
        )

    return output_path or video_path


def overlay_shapes(
    video_path: str,
    output_path: Optional[str] = None,
    num_shapes: int = 1,
    shape_type: Optional[str] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: Optional[int] = None,
    radius: Optional[float] = None,
    random_movement: bool = True,
    topleft: Optional[Tuple[float, float]] = None,
    bottomright: Optional[Tuple[float, float]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """
    Overlays random shapes onto a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param num_shapes: the number of shapes to add to each frame

    @param shape_type: specify if you would like circles or rectangles

    @param colors: list of (R, G, B) colors to sample from

    @param thickness: specify the thickness desired for the shapes

    @param random_movement: whether or not you want the shapes to randomly move
        around across the frame or to move across in a "linear" way

    @param topleft: specifying the boundary of shape region. The boundary are all
        floats [0, 1] representing the fraction w.r.t width/height

    @param bottomright: specifying the boundary of shape region. The boundary are all
        floats [0, 1] representing the fraction w.r.t width/height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    shapes_aug = ac.VideoDistractorByShapes(
        num_shapes,
        shape_type,
        colors,
        thickness,
        radius,
        random_movement,
        topleft,
        bottomright,
    )
    vdutils.apply_cv2_augmenter(shapes_aug, video_path, output_path, **kwargs)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="overlay_shapes", **func_kwargs
        )

    return output_path or video_path


def overlay_text(
    video_path: str,
    output_path: Optional[str] = None,
    text_len: int = 10,
    text_change_nth: Optional[int] = None,
    fonts: Optional[List[Tuple[Any, Optional[str]]]] = None,
    fontscales: Optional[Tuple[float, float]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: Optional[int] = None,
    random_movement: bool = False,
    topleft: Optional[Tuple[float, float]] = None,
    bottomright: Optional[Tuple[float, float]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """
    Overlays random text onto a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param text_len: length of string for randomized texts.

    @param text_change_nth: change random text every nth frame. None means
        using same text for all frames.

    @param fonts: list of fonts to sample from. Each font can be a cv2 fontFace,
        a PIL ImageFont, or a path to a PIL ImageFont file. Each font is coupled with
        a chars file (the second item in the tuple) - a path to a file which contains
        the characters associated with the given font. For example, non-western
        alphabets have different valid characters than the roman alphabet, and these
        must be specified in order to construct random valid text in that font. If the
        chars file path is None, the roman alphabet will be used.

    @param fontscales: 2-tuple of float (min_scale, max_scale).

    @param colors: list of (R, G, B) colors to sample from

    @param thickness: specify the thickness desired for the text.

    @param random_movement: whether or not you want the text to randomly move around
        across frame or to move across in a "linear" way

    @param topleft: specifying the boundary of text region. The boundary are all
        floats [0, 1] representing the fraction w.r.t width/height

    @param bottomright: specifying the boundary of text region. The boundary are all
        floats [0, 1] representing the fraction w.r.t width/height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    text_aug = ac.VideoDistractorByText(
        text_len,
        text_change_nth,
        fonts,
        fontscales,
        colors,
        thickness,
        random_movement,
        topleft,
        bottomright,
    )
    vdutils.apply_cv2_augmenter(text_aug, video_path, output_path, **kwargs)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="overlay_text", **func_kwargs
        )

    return output_path or video_path


def pad(
    video_path: str,
    output_path: Optional[str] = None,
    w_factor: float = 0.25,
    h_factor: float = 0.25,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Pads the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param w_factor: pad right and left with w_factor * frame width

    @param h_factor: pad bottom and top with h_factor * frame height

    @param color: RGB color of the padded margin

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    pad_aug = af.VideoAugmenterByPadding(w_factor, h_factor, color)
    pad_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="pad", **func_kwargs)

    return output_path or video_path


def perspective_transform_and_shake(
    video_path: str,
    output_path: Optional[str] = None,
    sigma: float = 50.0,
    shake_radius: float = 0.0,
    seed: Optional[int] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Apply a perspective transform to the video so it looks like it was taken
    as a photo from another device (e.g. taking a video from your phone of a
    video on a computer). Also has a shake factor to mimic the shakiness of
    someone holding a phone.

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param sigma: the standard deviation of the distribution of destination coordinates.
        The larger the sigma value, the more intense the transform

    @param shake_radius: determines the amount by which to "shake" the video; the larger
        the radius, the more intense the shake.

    @param seed: if provided, this will set the random seed to ensure consistency
        between runs

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    perspective_func = functools.partial(
        imaugs.perspective_transform, sigma=sigma, seed=seed
    )

    duration = float(helpers.get_video_info(video_path)["duration"])
    rng = np.random.RandomState(seed) if seed is not None else np.random

    def get_dx_dy(frame_number: int) -> Dict:
        u = math.sin(frame_number / duration * math.pi) ** 2
        return {
            "dx": u * rng.normal(0, shake_radius),
            "dy": u * rng.normal(0, shake_radius),
        }

    vdutils.apply_to_each_frame(perspective_func, video_path, output_path, get_dx_dy)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata,
            function_name="perspective_transform_and_shake",
            **func_kwargs,
        )

    return output_path or video_path


def pixelization(
    video_path: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Pixelizes the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param ratio: smaller values result in a more pixelated video, 1.0 indicates
        no change, and any value above one doesn't have a noticeable effect

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    assert ratio > 0, "Expected 'ratio' to be a positive number"
    video_info = helpers.get_video_info(video_path)
    width, height = video_info["width"], video_info["height"]

    output_path = output_path or video_path
    resize(video_path, output_path, height * ratio, width * ratio)
    resize(output_path, output_path, height, width)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="pixelization", **func_kwargs
        )

    return output_path or video_path


def remove_audio(
    video_path: str,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Removes the audio stream from a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    remove_audio_aug = af.VideoAugmenterByRemovingAudio()
    remove_audio_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="remove_audio", **func_kwargs
        )

    return output_path or video_path


def replace_with_color_frames(
    video_path: str,
    output_path: Optional[str] = None,
    offset_factor: float = 0.0,
    duration_factor: float = 1.0,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Replaces part of the video with frames of the specified color

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param offset_factor: start point of the replacement relative to the video
        duration (this parameter is multiplied by the video duration)

    @param duration_factor: the length of the replacement relative to the video
        duration (this parameter is multiplied by the video duration)

    @param color: RGB color of the replaced frames. Default color is black

    @param transition: optional transition configuration to apply between the clips

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    utils.validate_video_path(video_path)
    assert (
        0.0 <= offset_factor <= 1.0 and 0.0 <= duration_factor <= 1.0
    ), "Both offset & duration factors must be values in the range [0.0, 1.0]"

    func_kwargs = {
        **helpers.get_func_kwargs(metadata, locals(), video_path),
        "function_name": "replace_with_color_frames",
    }

    video_info = helpers.get_video_info(video_path)
    video_duration = float(video_info["duration"])
    width, height = video_info["width"], video_info["height"]

    offset = video_duration * offset_factor
    duration = video_duration * duration_factor
    output_path = output_path or video_path

    if duration == 0 or offset == video_duration:
        if output_path != video_path:
            shutil.copy(video_path, output_path)
        if metadata is not None:
            helpers.get_metadata(metadata=metadata, **func_kwargs)
        return output_path or video_path

    video_paths = []
    src_video_path_index = 0 if offset > 0 else 2
    with tempfile.TemporaryDirectory() as tmpdir:
        color_duration = (
            video_duration - offset if offset + duration >= video_duration else duration
        )
        color_path = os.path.join(tmpdir, "color_frames.mp4")
        helpers.create_color_video(color_path, color_duration, height, width, color)

        if helpers.has_audio_stream(video_path):
            audio_path = os.path.join(tmpdir, "audio.aac")
            helpers.extract_audio_to_file(video_path, audio_path)
            audio_swap(color_path, audio_path, offset=offset)

        if offset_factor == 0 and duration_factor == 1.0:
            shutil.copy(color_path, output_path)
            if metadata is not None:
                helpers.get_metadata(metadata=metadata, **func_kwargs)
            return output_path or video_path

        if offset > 0:
            before_path = os.path.join(tmpdir, "before.mp4")
            trim(video_path, before_path, end=offset)
            video_paths.append(before_path)

        video_paths.append(color_path)

        if offset + duration < video_duration:
            after_path = os.path.join(tmpdir, "after.mp4")
            trim(video_path, after_path, start=offset + duration)
            video_paths.append(after_path)

        concat(
            video_paths,
            output_path,
            src_video_path_index=src_video_path_index,
            transition=transition,
        )

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, **func_kwargs)

    return output_path or video_path


def resize(
    video_path: str,
    output_path: Optional[str] = None,
    height: Union[int, str] = "ih",
    width: Union[int, str] = "iw",
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Resizes a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param height: the height in which the video should be resized to. If not specified,
        the original video height will be used

    @param width: the width in which the video should be resized to. If not specified,
        the original video width will be used

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    resize_aug = af.VideoAugmenterByResize(height, width)
    resize_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="resize", **func_kwargs)

    return output_path or video_path


def rotate(
    video_path: str,
    output_path: Optional[str] = None,
    degrees: float = 15.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Rotates a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param degrees: expression for the angle by which to rotate the input video
        clockwise, expressed in degrees (supports negative values as well)

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    rotate_aug = af.VideoAugmenterByRotation(degrees)
    rotate_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="rotate", **func_kwargs)

    return output_path or video_path


def scale(
    video_path: str,
    output_path: Optional[str] = None,
    factor: float = 0.5,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Alters the resolution of a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param factor: the ratio by which the video should be downscaled or upscaled

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    scale_aug = af.VideoAugmenterByResolution(factor)
    scale_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="scale", **func_kwargs)

    return output_path or video_path


def shift(
    video_path: str,
    output_path: Optional[str] = None,
    x_factor: float = 0.0,
    y_factor: float = 0.0,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Shifts the original frame position from the center by a vector
    (width * x_factor, height * y_factor) and pads the rest with a
    colored margin

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param x_factor: the horizontal amount that the video should be shifted,
        relative to the width of the video

    @param y_factor: the vertical amount that the video should be shifted,
        relative to the height of the video

    @param color: RGB color of the margin generated by the shift. Default color is black

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    utils.validate_video_path(video_path)
    video_info = helpers.get_video_info(video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        background_path = os.path.join(tmpdir, "background.mp4")
        helpers.create_color_video(
            background_path,
            float(video_info["duration"]),
            video_info["height"],
            video_info["width"],
            color,
        )

        overlay(
            background_path,
            video_path,
            output_path,
            x_factor=x_factor,
            y_factor=y_factor,
            use_overlay_audio=True,
        )

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="shift", **func_kwargs)

    return output_path or video_path


def time_crop(
    video_path: str,
    output_path: Optional[str] = None,
    offset_factor: float = 0.0,
    duration_factor: float = 1.0,
    minimum_duration: float = 0.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Crops the video using the specified offset and duration factors

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param offset_factor: start point of the crop relative to the video duration
        (this parameter is multiplied by the video duration)

    @param duration_factor: the length of the crop relative to the video duration
        (this parameter is multiplied by the video duration)

    @param minimum_duration: the minimum duration of a segment selected

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    time_crop_aug = af.VideoAugmenterByTrim(
        offset_factor=offset_factor,
        duration_factor=duration_factor,
        minimum_duration=minimum_duration,
    )
    time_crop_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="time_crop", **func_kwargs
        )

    return output_path or video_path


def time_decimate(
    video_path: str,
    output_path: Optional[str] = None,
    start_offset_factor: float = 0.0,
    on_factor: float = 0.2,
    off_factor: float = 0.5,
    transition: Optional[af.TransitionConfig] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Removes evenly sized (off) chunks, and concatenates evenly spaced (on)
    chunks from the video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param start_offset_factor: relative to the video duration; the offset
        at which to start taking "on" segments

    @param on_factor: relative to the video duration; the amount of time each
        "on" video chunk should be

    @param off_factor: relative to the "on" duration; the amount of time each
        "off" video chunk should be

    @param transition: optional transition configuration to apply between the clips

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    assert (
        0 <= start_offset_factor < 1
    ), f"start_offset_factor value {start_offset_factor} must be in the range [0, 1)"
    assert 0 < on_factor <= 1, "on_factor must be a value in the range (0, 1]"
    assert 0 <= off_factor <= 1, "off_factor must be a value in the range [0, 1]"

    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
    local_path = utils.pathmgr.get_local_path(video_path)
    utils.validate_video_path(local_path)

    video_info = helpers.get_video_info(local_path)
    _, video_ext = os.path.splitext(local_path)

    duration = float(video_info["duration"])
    start_offset = duration * start_offset_factor
    on_segment = duration * on_factor
    off_segment = on_segment * off_factor

    subclips = []
    n = int((duration - start_offset) / (on_segment + off_segment))

    # let a = on_segment and b = off_segment
    # subclips: 0->a, a+b -> 2*a + b, 2a+2b -> 3a+2b, .., na+nb -> (n+1)a + nb
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n):
            clip_path = os.path.join(tmpdir, f"{i}{video_ext}")
            trim(
                video_path,
                clip_path,
                start=start_offset + i * on_segment + i * off_segment,
                end=min(
                    duration, start_offset + (i + 1) * on_segment + i * off_segment
                ),
            )
            subclips.append(clip_path)

        # Skip concatenation if only 1 clip.
        if n > 1:
            concat(
                subclips,
                output_path,
                transition=transition,
            )
        else:
            if output_path is not None:
                shutil.copy(subclips[0], output_path)
            else:
                shutil.copy(subclips[0], local_path)

    if metadata is not None:
        helpers.get_metadata(
            metadata=metadata, function_name="time_decimate", **func_kwargs
        )

    return output_path or video_path


def trim(
    video_path: str,
    output_path: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Trims the video using the specified start and end parameters

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param start: starting point in seconds of when the trimmed video should start.
        If None, start will be 0

    @param end: ending point in seconds of when the trimmed video should end.
        If None, the end will be the duration of the video

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    trim_aug = af.VideoAugmenterByTrim(start=start, end=end)
    trim_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="trim", **func_kwargs)

    return output_path or video_path


def vflip(
    video_path: str,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Vertically flips a video

    @param video_path: the path to the video to be augmented

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    vflip_aug = af.VideoAugmenterByVFlip()
    vflip_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="vflip", **func_kwargs)

    return output_path or video_path


def vstack(
    video_path: str,
    second_video_path: str,
    output_path: Optional[str] = None,
    use_second_audio: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Vertically stacks two videos

    @param video_path: the path to the video that will be stacked on top

    @param second_video_path: the path to the video that will be stacked on the bottom

    @param output_path: the path in which the resulting video will be stored.
        If not passed in, the original video file will be overwritten

    @param use_second_audio: if set to True, the audio of the bottom video will be used
        instead of the top's

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, fps, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @returns: the path to the augmented video
    """
    func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)

    vstack_aug = af.VideoAugmenterByStack(second_video_path, use_second_audio, "vstack")
    vstack_aug.add_augmenter(video_path, output_path)

    if metadata is not None:
        helpers.get_metadata(metadata=metadata, function_name="vstack", **func_kwargs)

    return output_path or video_path
