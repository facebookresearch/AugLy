#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional, Tuple

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByResize(BaseFFMPEGAugmenter):
    def __init__(self, height: Optional[int], width: Optional[int]):
        assert height is None or height > 0, "Height must be set to None or be positive"
        assert width is None or width > 0, "Width must be set to None or be positive"

        self.new_h = height or "ih"
        self.new_w = width or "iw"

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Resizes the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return (
            in_stream.video.filter(
                "scale", **{"width": self.new_w, "height": self.new_h}
            ).filter("pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"}),
            {},
        )
