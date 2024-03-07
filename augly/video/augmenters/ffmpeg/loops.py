#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByLoops(BaseVidgearFFMPEGAugmenter):
    def __init__(self, num_loops: int):
        assert type(num_loops) == int, "Number of loops must be an integer"
        assert num_loops >= 0, "Number of loops cannot be a negative number"
        self.num_loops = num_loops

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Loops the video `num_loops` times

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return [
            "-y",
            "-stream_loop",
            str(self.num_loops),
            "-i",
            video_path,
            "-c:a",
            "copy",
            *self.output_fmt(output_path),
        ]
