#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByNoise(BaseVidgearFFMPEGAugmenter):
    def __init__(self, level: int):
        assert 0 <= level <= 100, "Level must be a value in the range [0, 100]"
        self.level = level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Adds noise to the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"boxblur=lr=1.2,noise=c0s={self.level}:allf=t"], output_path
        )
