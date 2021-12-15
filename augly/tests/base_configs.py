#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from augly.utils import TEST_URI


@dataclass
class BaseAugConfig:
    output_dir: str
    input_dir: str
    input_files: Sequence[str]
    input_file_index: int = 0

    def get_input_path(self, input_file_index: Optional[int] = None) -> Tuple[str, str]:
        if input_file_index is None:
            input_file_index = self.input_file_index

        filename = self.input_files[input_file_index]
        filepath = os.path.join(self.input_dir, filename)
        return filepath, filename

    def get_output_path(self, filename: str, prefix: str = "") -> str:
        aug_filename = f"aug_{prefix}{filename}"
        output_path = os.path.join(self.output_dir, aug_filename)
        return output_path


@dataclass
class ImageAugConfig(BaseAugConfig):
    output_dir: str = os.path.join(TEST_URI, "image", "outputs")
    input_dir: str = os.path.join(TEST_URI, "image", "inputs")
    input_files: Sequence[str] = ("dfdc_1.jpg", "dfdc_2.jpg", "dfdc_3.jpg")


@dataclass
class VideoAugConfig(BaseAugConfig):
    output_dir: str = os.path.join(TEST_URI, "video", "outputs")
    input_dir: str = os.path.join(TEST_URI, "video", "inputs")
    input_files: Sequence[str] = ("input_1.mp4", "input_2.mp4", "input_3.mp4")
    input_audio_file: str = os.path.join(TEST_URI, "video", "inputs", "input_1.aac")


@dataclass
class AudioAugConfig(BaseAugConfig):
    output_dir: str = os.path.join(TEST_URI, "audio", "outputs")
    input_dir: str = os.path.join(TEST_URI, "audio", "inputs")
    input_files: Sequence[str] = ("vad-go-mono-32000.wav", "vad-go-stereo-44100.wav")
