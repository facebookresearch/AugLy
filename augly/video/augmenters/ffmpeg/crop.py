#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByCrop(BaseFFMPEGAugmenter):
    def __init__(self, left: float, top: float, right: float, bottom: float):
        assert 0.0 <= left <= 1.0, "Left must be a value in the range [0.0, 1.0]"
        assert 0.0 <= top <= 1.0, "Top must be a value in the range [0.0, 1.0]"
        assert left < right <= 1.0, "Right must be a value in the range (left, 1.0]"
        assert top < bottom <= 1.0, "Bottom must be a value in the range (top, 1.0]"

        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Crops the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])

        x1 = int(video_info["width"] * self.left)
        y1 = int(video_info["height"] * self.top)
        width = int(video_info["width"] * (self.right - self.left))
        height = int(video_info["height"] * (self.bottom - self.top))

        return in_stream.video.crop(x1, y1, width, height), {}
