#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info


class VideoAugmenterByCrop(BaseVidgearFFMPEGAugmenter):
    def __init__(self, left: float, top: float, right: float, bottom: float):
        assert 0.0 <= left <= 1.0, "Left must be a value in the range [0.0, 1.0]"
        assert 0.0 <= top <= 1.0, "Top must be a value in the range [0.0, 1.0]"
        assert left < right <= 1.0, "Right must be a value in the range (left, 1.0]"
        assert top < bottom <= 1.0, "Bottom must be a value in the range (top, 1.0]"

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Crops the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
        """
        video_info = get_video_info(video_path)

        x1 = int(video_info["width"] * self.left)
        y1 = int(video_info["height"] * self.top)
        width = int(video_info["width"] * (self.right - self.left))
        height = int(video_info["height"] * (self.bottom - self.top))

        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            f"crop=w={width}:h={height}:x={x1}:y={y1}",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
