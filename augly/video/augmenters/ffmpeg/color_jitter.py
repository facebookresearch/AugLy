#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByColorJitter(BaseFFMPEGAugmenter):
    def __init__(
        self, brightness_level: float, contrast_level: float, saturation_level: float
    ):
        assert (
            -1.0 <= brightness_level <= 1.0
        ), "Brightness factor must be a value in the range [-1.0, 1.0]"
        assert (
            -1000.0 <= contrast_level <= 1000.0
        ), "Contrast factor must be a value in the range [-1000, 1000]"
        assert (
            0.0 <= saturation_level <= 3.0
        ), "Saturation factor must be a value in the range [0.0, 3.0]"

        self.brightness_level = brightness_level
        self.contrast_level = contrast_level
        self.saturation_level = saturation_level

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Color jitters the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        return (
            in_stream.video.filter(
                "eq",
                **{
                    "brightness": self.brightness_level,
                    "contrast": self.contrast_level,
                    "saturation": self.saturation_level,
                }
            ),
            {},
        )
