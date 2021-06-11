#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByQuality(BaseFFMPEGAugmenter):
    def __init__(self, quality: int):
        assert 0 <= quality <= 51, "Quality must be a value in the range [0, 51]"
        self.quality = quality

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Alters the quality level of the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return in_stream.video, {"crf": self.quality}
