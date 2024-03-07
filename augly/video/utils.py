#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import os
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Union

from augly import utils
from augly.video import helpers as helpers
from augly.video.augmenters import cv2 as ac


"""
Utility Functions: Augmentation Application Functions
- For FFMPEG-Based Functions
- For CV2-Based Functions
- For Applying Image Functions to Each Frame
"""


def apply_to_each_frame(
    img_aug_function: functools.partial,
    video_path: str,
    output_path: Optional[str],
    frame_func: Optional[Callable[[int], Dict]] = None,
) -> None:
    video_path, output_path = helpers.validate_input_and_output_paths(
        video_path, output_path
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = os.path.join(tmpdir, "video_frames")
        os.mkdir(frame_dir)
        helpers.extract_frames_to_dir(video_path, frame_dir)

        for i, frame_file in enumerate(os.listdir(frame_dir)):
            frame_path = os.path.join(frame_dir, frame_file)
            kwargs = frame_func(i) if frame_func is not None else {}
            img_aug_function(frame_path, output_path=frame_path, **kwargs)

        audio_path = None
        if helpers.has_audio_stream(video_path):
            audio_path = os.path.join(tmpdir, "audio.aac")
            helpers.extract_audio_to_file(video_path, audio_path)

        helpers.combine_frames_and_audio_to_file(
            os.path.join(frame_dir, "raw_frame*.jpg"),
            audio_path,
            output_path,
            helpers.get_video_fps(video_path) or utils.DEFAULT_FRAME_RATE,
        )


def apply_to_frames(
    img_aug_function: functools.partial,
    video_path: str,
    second_video_path: str,
    output_path: Optional[str],
    use_second_audio: bool = False,
) -> None:
    video_path, output_path = helpers.validate_input_and_output_paths(
        video_path, output_path
    )
    second_video_path = helpers.get_local_path(second_video_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = os.path.join(tmpdir, "video_frames")
        os.mkdir(frame_dir)
        helpers.extract_frames_to_dir(video_path, frame_dir)

        second_frame_dir = os.path.join(tmpdir, "second_video_frames")
        os.mkdir(second_frame_dir)
        helpers.extract_frames_to_dir(second_video_path, second_frame_dir)

        for frame_file, second_frame_file in zip(
            os.listdir(frame_dir), os.listdir(second_frame_dir)
        ):
            img_aug_function(
                os.path.join(frame_dir, frame_file),
                os.path.join(second_frame_dir, second_frame_file),
                output_path=os.path.join(frame_dir, frame_file),
            )

        audio_path = None
        if not use_second_audio and helpers.has_audio_stream(video_path):
            audio_path = os.path.join(tmpdir, "audio.aac")
            helpers.extract_audio_to_file(video_path, audio_path)
        elif use_second_audio and helpers.has_audio_stream(second_video_path):
            audio_path = os.path.join(tmpdir, "audio.aac")
            helpers.extract_audio_to_file(second_video_path, audio_path)

        helpers.combine_frames_and_audio_to_file(
            os.path.join(frame_dir, "raw_frame*.jpg"),
            audio_path,
            output_path,
            helpers.get_video_fps(video_path) or utils.DEFAULT_FRAME_RATE,
        )


def apply_cv2_augmenter(
    distractor: ac.BaseCV2Augmenter,
    video_path: str,
    output_path: Optional[str],
    **kwargs,
) -> None:
    video_path, output_path = helpers.validate_input_and_output_paths(
        video_path, output_path
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        video_tmp_path = os.path.join(tmpdir, os.path.basename(video_path))
        shutil.copyfile(video_path, video_tmp_path)

        fps = helpers.get_video_fps(video_tmp_path) or utils.DEFAULT_FRAME_RATE
        aug_frame_temp_dir = distractor.augment(video_tmp_path, fps, **kwargs)

        audio_path = None
        if helpers.has_audio_stream(video_path):
            audio_path = os.path.join(tmpdir, "audio.aac")
            helpers.extract_audio_to_file(video_path, audio_path)

        helpers.combine_frames_and_audio_to_file(
            os.path.join(aug_frame_temp_dir, "raw_frame*.jpg"),
            audio_path,
            video_tmp_path,
            fps,
        )
        shutil.move(video_tmp_path, output_path)
        shutil.rmtree(aug_frame_temp_dir)


def get_image_kwargs(imgs_dir: str) -> Dict[str, Optional[Union[List[str], str]]]:
    return {"imgs_dir": imgs_dir, "imgs_files": utils.pathmgr.ls(imgs_dir)}
