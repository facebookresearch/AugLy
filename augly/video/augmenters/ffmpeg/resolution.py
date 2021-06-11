#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByResolution(BaseFFMPEGAugmenter):
    def __init__(self, resolution: float):
        assert (
            isinstance(resolution, (int, float)) and resolution > 0.0
        ), "Invalid resolution: scale factor must be positive"

        self.resolution = resolution

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Alters the resolution of the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return (
            in_stream.video.filter(
                "scale", f"iw*{self.resolution}", f"ih*{self.resolution}"
            ).filter("pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"}),
            {},
        )
