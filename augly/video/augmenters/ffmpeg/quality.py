#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByQuality(BaseVidgearFFMPEGAugmenter):
    def __init__(self, quality: int):
        assert 0 <= quality <= 51, "Quality must be a value in the range [0, 51]"
        self.quality = quality

    def get_command(
        self, video_path: str, output_path: Optional[str] = None
    ) -> List[str]:
        """
        Alters the quality level of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-crf",
            f"{self.quality}",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]
        return command
