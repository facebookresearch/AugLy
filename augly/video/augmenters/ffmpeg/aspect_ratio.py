#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import List, Union

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info


class VideoAugmenterByAspectRatio(BaseVidgearFFMPEGAugmenter):
    def __init__(self, ratio: Union[float, str]):
        assert (isinstance(ratio, str) and len(ratio.split(":")) == 2) or (
            isinstance(ratio, (int, float)) and ratio > 0
        ), "Aspect ratio must be a valid string ratio or a positive number"
        self.aspect_ratio = ratio

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the sample (sar) & display (dar) aspect ratios of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        video_info = get_video_info(video_path)

        area = int(video_info["width"]) * int(video_info["height"])
        if isinstance(self.aspect_ratio, float):
            aspect_ratio = float(self.aspect_ratio)
        else:
            num, denom = [int(x) for x in str(self.aspect_ratio).split(":")]
            aspect_ratio = num / denom

        new_w = int(math.sqrt(area * aspect_ratio))
        new_h = int(area / new_w)

        filters = [
            f"scale=width={new_w}:height={new_h},"
            + "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2,"
            + f"setsar=ratio={self.aspect_ratio},"
            + f"setdar=ratio={self.aspect_ratio}",
        ]

        return self.standard_filter_fmt(video_path, filters, output_path)
