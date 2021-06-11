#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple, Dict

import ffmpeg  # @manual
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import has_audio_stream
from ffmpeg.nodes import FilterableStream


class VideoAugmenterBySpeed(BaseFFMPEGAugmenter):
    def __init__(self, factor: float):
        assert factor > 0, "Factor must be greater than zero"
        self.factor = factor

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Changes the speed of the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video = in_stream.video.setpts(f"{1/self.factor}*PTS")
        if has_audio_stream(kwargs["video_path"]):
            audio = in_stream.audio.filter_("atempo", self.factor)
            video = ffmpeg.concat(video, audio, v=1, a=1)
        return video, {}
