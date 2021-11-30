#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseFFMPEGAugmenter


class VideoAugmenterByHFlip(BaseFFMPEGAugmenter):
    def add_augmenter(self, video_path, output_path, **kwargs) -> List[str]:
        """
        Horizontally flips the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            "hflip",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
