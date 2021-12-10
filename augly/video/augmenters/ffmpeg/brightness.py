#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByBrightness(BaseVidgearFFMPEGAugmenter):
    def __init__(self, level: float):
        assert -1.0 <= level <= 1.0, "Level must be a value in the range [-1.0, 1.0]"
        self.level = level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Changes the brightness level of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            f"eq=brightness={self.level}",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
