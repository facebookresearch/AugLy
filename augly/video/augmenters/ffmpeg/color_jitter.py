#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByColorJitter(BaseVidgearFFMPEGAugmenter):
    def __init__(
        self, brightness_level: float, contrast_level: float, saturation_level: float
    ):
        assert (
            -1.0 <= brightness_level <= 1.0
        ), "Brightness factor must be a value in the range [-1.0, 1.0]"
        assert (
            -1000.0 <= contrast_level <= 1000.0
        ), "Contrast factor must be a value in the range [-1000, 1000]"
        assert (
            0.0 <= saturation_level <= 3.0
        ), "Saturation factor must be a value in the range [0.0, 3.0]"

        self.brightness_level = brightness_level
        self.contrast_level = contrast_level
        self.saturation_level = saturation_level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Color jitters the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        filters = [
            f"eq=brightness={self.brightness_level}"
            + f":contrast={self.contrast_level}"
            + f":saturation={self.saturation_level}"
        ]

        return self.standard_filter_fmt(video_path, filters, output_path)
