#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByNoise(BaseVidgearFFMPEGAugmenter):
    def __init__(self, level: int):
        assert 0 <= level <= 100, "Level must be a value in the range [0, 100]"
        self.level = level

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Adds noise to the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            "boxblur=lr=1.2," + f"noise=c0s={self.level}:allf=t",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
