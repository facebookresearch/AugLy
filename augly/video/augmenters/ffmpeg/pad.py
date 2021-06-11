#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

from augly.utils import validate_rgb_color
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByPadding(BaseFFMPEGAugmenter):
    def __init__(self, w_factor: float, h_factor: float, color: Tuple[int, int, int]):
        assert w_factor >= 0, "w_factor cannot be a negative number"
        assert h_factor >= 0, "h_factor cannot be a negative number"
        validate_rgb_color(color)

        self.w_factor = w_factor
        self.h_factor = h_factor
        self.hex_color = "%02x%02x%02x" % color

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Adds padding to the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])

        left = int(video_info["width"] * self.w_factor)
        top = int(video_info["height"] * self.h_factor)

        return (
            in_stream.video.filter(
                "pad",
                **{
                    "width": f"iw+{left*2}",
                    "height": f"ih+{top*2}",
                    "x": left,
                    "y": top,
                    "color": self.hex_color,
                },
            ),
            {},
        )
