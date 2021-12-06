#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

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

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
        """
        command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps=fps={self.fps}:round=up",
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]

        return command
