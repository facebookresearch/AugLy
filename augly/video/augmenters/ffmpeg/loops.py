#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter


class VideoAugmenterByLoops(BaseVidgearFFMPEGAugmenter):
    def __init__(self, num_loops: int):
        assert type(num_loops) == int, "Number of loops must be an integer"
        assert num_loops >= 0, "Number of loops cannot be a negative number"
        self.num_loops = num_loops

    def get_command(
        self, video_path: str, output_path: Optional[str] = None, **kwargs
    ) -> List[str]:
        """
        Loops the video `num_loops` times

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
        """
        command = [
            "-y",
            "-stream_loop",
            str(self.num_loops),
            "-i",
            video_path,
            "-c:a",
            "copy",
            "-preset",
            "ultrafast",
            output_path,
        ]
        return command
