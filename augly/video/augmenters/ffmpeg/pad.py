#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Tuple

from augly.utils import validate_rgb_color
from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info


class VideoAugmenterByPadding(BaseVidgearFFMPEGAugmenter):
    def __init__(self, w_factor: float, h_factor: float, color: Tuple[int, int, int]):
        assert w_factor >= 0, "w_factor cannot be a negative number"
        assert h_factor >= 0, "h_factor cannot be a negative number"
        validate_rgb_color(color)

        self.w_factor = w_factor
        self.h_factor = h_factor
        self.hex_color = "%02x%02x%02x" % color

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Adds padding to the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        video_info = get_video_info(video_path)

        left = int(video_info["width"] * self.w_factor)
        top = int(video_info["height"] * self.h_factor)

        filters = [
            f"pad=width={left*2}+iw:height={top*2}+ih"
            + f":x={left}:y={top}:color={self.hex_color}"
        ]

        return self.standard_filter_fmt(video_path, filters, output_path)
