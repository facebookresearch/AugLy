#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByResize(BaseVidgearFFMPEGAugmenter):
    def __init__(self, height: Optional[int], width: Optional[int]):
        assert height is None or height > 0, "Height must be set to None or be positive"
        assert width is None or width > 0, "Width must be set to None or be positive"

        self.new_h = height or "ih"
        self.new_w = width or "iw"

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Resizes the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        filters = [
            f"scale=height:{self.new_h}:width={self.new_w},"
            + "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
        ]

        return self.standard_filter_fmt(video_path, filters, output_path)
