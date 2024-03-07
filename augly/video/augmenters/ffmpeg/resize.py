#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Union

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByResize(BaseVidgearFFMPEGAugmenter):
    def __init__(self, height: Union[int, str] = "ih", width: Union[int, str] = "iw"):
        """
        Constructor. See https://trac.ffmpeg.org/wiki/Scaling for height and width options.

        @param height: height specification. Defaults to input if not specified.

        @param width: width specification. Defaults to input width if not specified.
        """
        self.new_h = height
        self.new_w = width

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
