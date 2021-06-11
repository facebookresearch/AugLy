#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByNoise(BaseFFMPEGAugmenter):
    def __init__(self, level: int):
        assert 0 <= level <= 100, "Level must be a value in the range [0, 100]"
        self.level = level

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Adds noise to the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return (
            in_stream.video.filter("boxblur", **{"lr": 1.2}).filter(
                "noise", **{"c0s": self.level, "allf": "t"}
            ),
            {},
        )
