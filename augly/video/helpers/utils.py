#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
from augly import utils
from augly.utils.ffmpeg import FFMPEG_PATH
from augly.video import helpers


DEFAULT_FRAME_RATE = 10


def create_color_video(
    output_path: str,
    duration: float,
    height: int,
    width: int,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
) -> None:
    """
    Creates a video with frames of the specified color

    @param output_path: the path in which the resulting video will be stored

    @param duration: how long the video should be, in seconds

    @param height: the desired height of the video to be generated

    @param width: the desired width of the video to be generated

    @param color: RGB color of the video. Default color is black
    """
    utils.validate_output_path(output_path)
    assert duration > 0, "Duration of the video must be a positive value"
    assert height > 0, "Height of the video must be a positive value"
    assert width > 0, "Width of the video must be a positive value"

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "image.png")
        color_frame = np.full((height, width, 3), color[::-1])
        cv2.imwrite(image_path, color_frame)
        create_video_from_image(output_path, image_path, duration)


def create_video_from_image(output_path: str, image_path: str, duration: float) -> None:
    """
    Creates a video with all frames being the image provided

    @param output_path: the path in which the resulting video will be stored

    @param image_path: the path to the image to use to create the video

    @param duration: how long the video should be, in seconds
    """
    utils.validate_output_path(output_path)
    utils.validate_image_path(image_path)
    assert duration > 0, "Duration of the video must be a positive value"

    im_stream = ffmpeg.input(image_path, stream_loop=-1)
    video = im_stream.filter("framerate", utils.DEFAULT_FRAME_RATE).filter(
        "pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"}
    )

    silent_audio_path = utils.pathmgr.get_local_path(utils.SILENT_AUDIO_PATH)
    audio = ffmpeg.input(silent_audio_path, stream_loop=math.ceil(duration)).audio
    output = ffmpeg.output(video, audio, output_path, pix_fmt="yuv420p", t=duration)
    output.overwrite_output().run(cmd=FFMPEG_PATH)


def get_local_path(video_path: str) -> str:
    return utils.pathmgr.get_local_path(video_path)


def identity_function(
    video_path: str,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> str:
    video_path, output_path = validate_input_and_output_paths(video_path, output_path)

    if output_path is not None and output_path != video_path:
        shutil.copy(video_path, output_path)
    if metadata is not None:
        func_kwargs = helpers.get_func_kwargs(metadata, locals(), video_path)
        helpers.get_metadata(
            metadata=metadata, function_name="identity_function", **func_kwargs
        )
    return output_path or video_path


def validate_input_and_output_paths(
    video_path: str, output_path: Optional[str]
) -> Tuple[str, str]:
    local_video_path = get_local_path(video_path)
    utils.validate_video_path(local_video_path)

    if output_path is None:
        assert (
            video_path == local_video_path
        ), "If using a nonlocal input path, you must specify an output path"

    output_path = output_path or video_path
    utils.validate_output_path(output_path)

    return local_video_path, output_path
