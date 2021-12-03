#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByColorJitter(BaseVidgearFFMPEGAugmenter):
    def __init__(
        self, brightness_level: float, contrast_level: float, saturation_level: float
    ):
        assert (
            -1.0 <= brightness_level <= 1.0
        ), "Brightness factor must be a value in the range [-1.0, 1.0]"
        assert (
            -1000.0 <= contrast_level <= 1000.0
        ), "Contrast factor must be a value in the range [-1000, 1000]"
        assert (
            0.0 <= saturation_level <= 3.0
        ), "Saturation factor must be a value in the range [0.0, 3.0]"

        self.brightness_level = brightness_level
        self.contrast_level = contrast_level
        self.saturation_level = saturation_level

    def get_command(
        self, video_path: str, output_path: Optional[str] = None, **kwargs
    ) -> List[str]:
        """
        Color jitters the video

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
            f"eq=brightness={self.brightness_level}:contrast={self.contrast_level}:saturation={self.saturation_level}",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]
        return command
