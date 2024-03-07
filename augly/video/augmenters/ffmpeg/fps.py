#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByFPSChange(BaseVidgearFFMPEGAugmenter):
    def __init__(self, fps: int):
        assert fps > 0, "FPS must be greater than zero"
        self.fps = fps

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the frame rate of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        return self.standard_filter_fmt(
            video_path, [f"fps=fps={self.fps}:round=up"], output_path
        )
