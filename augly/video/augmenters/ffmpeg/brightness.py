#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByBrightness(BaseVidgearFFMPEGAugmenter):
    def __init__(self, level: float):
        assert -1.0 <= level <= 1.0, "Level must be a value in the range [-1.0, 1.0]"
        self.level = level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the brightness level of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"eq=brightness={self.level}"], output_path
        )
