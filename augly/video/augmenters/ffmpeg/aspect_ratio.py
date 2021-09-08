#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Dict, Tuple, Union

from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByAspectRatio(BaseFFMPEGAugmenter):
    def __init__(self, ratio: Union[float, str]):
        assert (
            (isinstance(ratio, str) and len(ratio.split(":")) == 2)
            or (isinstance(ratio, (int, float)) and ratio > 0)
        ), "Aspect ratio must be a valid string ratio or a positive number"
        self.aspect_ratio = ratio

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Changes the sample (sar) & display (dar) aspect ratios of the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])

        area = int(video_info["width"]) * int(video_info["height"])
        if isinstance(self.aspect_ratio, float):
            aspect_ratio = float(self.aspect_ratio)
        else:
            num, denom = [int(x) for x in str(self.aspect_ratio).split(":")]
            aspect_ratio = num / denom

        new_w = int(math.sqrt(area * aspect_ratio))
        new_h = int(area / new_w)

        return (
            in_stream.video.filter("scale", **{"width": new_w, "height": new_h})
            .filter("pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"})
            .filter("setsar", **{"ratio": self.aspect_ratio})
            .filter("setdar", **{"ratio": self.aspect_ratio}),
            {},
        )
