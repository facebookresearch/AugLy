#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByBlur(BaseVidgearFFMPEGAugmenter):
    def __init__(self, sigma: float):
        assert sigma >= 0, "Sigma cannot be a negative number"
        self.sigma = sigma

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Blurs the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"gblur={self.sigma}"], output_path
        )
