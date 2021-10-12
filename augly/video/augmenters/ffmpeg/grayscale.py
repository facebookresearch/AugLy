#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

from augly.video.augmenters.ffmpeg.base_augmenter import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByGrayscale(BaseFFMPEGAugmenter):
    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Changes the video to be grayscale

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return in_stream.video.filter("hue", **{"s": 0}), {}
