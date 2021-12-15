#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple

import magic
from augly.utils.io import pathmgr


def is_content_type(filename: str, content_type: str) -> bool:
    file_type = magic.from_file(filename, mime=True).lower()
    return content_type in file_type


def is_audio_file(filename: str) -> bool:
    return is_content_type(filename, "audio")


def is_image_file(filename: str) -> bool:
    return is_content_type(filename, "image")


def is_video_file(filename: str) -> bool:
    return is_content_type(filename, "video")


def validate_path(file_path: str) -> None:
    correct_type = type(file_path) == str
    path_exists = pathmgr.exists(file_path)
    assert correct_type and path_exists, f"Path is invalid: {file_path}"


def validate_audio_path(audio_path: str) -> None:
    validate_path(audio_path)

    # since `librosa` can extract audio from audio and video
    # paths, we check for both here
    assert is_audio_file(audio_path) or is_video_file(
        audio_path
    ), f"Audio path invalid: {audio_path}"


def validate_image_path(image_path: str) -> None:
    validate_path(image_path)
    assert is_image_file(image_path), f"Image path invalid: {image_path}"


def validate_video_path(video_path: str) -> None:
    validate_path(video_path)
    assert is_video_file(video_path), f"Video path invalid: {video_path}"


def validate_output_path(output_path: str) -> None:
    correct_type = type(output_path) == str
    dir_exists = pathmgr.exists(os.path.dirname(output_path))
    assert correct_type and dir_exists, f"Output path invalid: {output_path}"


def validate_rgb_color(color: Tuple[int, int, int]) -> None:
    correct_len = len(color) == 3
    correct_values = all(0 <= c <= 255 for c in color)
    assert correct_len and correct_values, "Invalid RGB color specified"
