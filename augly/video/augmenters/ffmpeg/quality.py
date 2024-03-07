#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByQuality(BaseVidgearFFMPEGAugmenter):
    def __init__(self, quality: int):
        assert 0 <= quality <= 51, "Quality must be a value in the range [0, 51]"
        self.quality = quality

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Alters the quality level of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return [
            *self.input_fmt(video_path),
            "-c:v",
            "libx264",
            "-crf",
            f"{self.quality}",
            "-c:a",
            "copy",
            *self.output_fmt(output_path),
        ]
