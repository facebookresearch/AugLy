#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByResolution(BaseVidgearFFMPEGAugmenter):
    def __init__(self, resolution: float):
        assert (
            isinstance(resolution, (int, float)) and resolution > 0.0
        ), "Invalid resolution: scale factor must be positive"

        self.resolution = resolution

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Alters the resolution of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        filters = [
            f"scale=height:ih*{self.resolution}:width=iw*{self.resolution},"
            + "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"
        ]

        return self.standard_filter_fmt(video_path, filters, output_path)
