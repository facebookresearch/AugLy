#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

import ffmpeg  # @manual
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByLoops(BaseFFMPEGAugmenter):
    def __init__(self, num_loops: int):
        assert type(num_loops) == int, "Number of loops must be an integer"
        assert num_loops >= 0, "Number of loops cannot be a negative number"
        self.num_loops = num_loops

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Loops the video `num_loops` times

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        looped = ffmpeg.input(kwargs["video_path"], stream_loop=self.num_loops)
        return ffmpeg.concat(looped.video, looped.audio, v=1, a=1, n=1), {}
