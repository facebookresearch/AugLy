#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Tuple, Dict

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByRotation(BaseFFMPEGAugmenter):
    def __init__(self, degrees: float):
        assert isinstance(degrees, (float, int)), "Expected 'degrees' to be a number"
        self.degrees = degrees

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Rotates the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return in_stream.video.filter("rotate", self.degrees * (math.pi / 180)), {}
