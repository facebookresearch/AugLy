#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByHFlip(BaseVidgearFFMPEGAugmenter):
    def get_command(self, video_path: str, output_path: str, **kwargs) -> List[str]:
        """
        Horizontally flips the video

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
            "hflip",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
