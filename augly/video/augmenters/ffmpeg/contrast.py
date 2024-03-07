#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByContrast(BaseVidgearFFMPEGAugmenter):
    def __init__(self, level: float):
        assert (
            -1000.0 <= level <= 1000.0
        ), "Level must be a value in the range [-1000, 1000]"
        self.level = level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the contrast level of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"eq=contrast={self.level}"], output_path
        )
