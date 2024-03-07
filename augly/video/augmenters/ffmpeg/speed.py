#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterBySpeed(BaseVidgearFFMPEGAugmenter):
    def __init__(self, factor: float):
        assert factor > 0, "Factor must be greater than zero"
        self.factor = factor

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the speed of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return [
            *self.input_fmt(video_path),
            "-vf",
            f"setpts={1/self.factor}*PTS",
            "-filter:a",
            f"atempo={self.factor}",
            *self.output_fmt(output_path),
        ]
