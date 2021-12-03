#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByBlur(BaseVidgearFFMPEGAugmenter):
    def __init__(self, sigma: float):
        assert sigma >= 0, "Sigma cannot be a negative number"
        self.sigma = sigma

    def get_command(
        self, video_path: str, output_path: Optional[str] = None, **kwargs
    ) -> List[str]:
        """
        Blurs the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            f"gblur={self.sigma}",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]
        return command
