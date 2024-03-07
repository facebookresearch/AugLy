#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByRotation(BaseVidgearFFMPEGAugmenter):
    def __init__(self, degrees: float):
        assert isinstance(degrees, (float, int)), "Expected 'degrees' to be a number"
        self.degrees = degrees

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Rotates the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"rotate={self.degrees * (math.pi / 180)}"], output_path
        )
