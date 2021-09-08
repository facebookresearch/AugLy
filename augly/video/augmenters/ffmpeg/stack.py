#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

import ffmpeg  # @manual
from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByStack(BaseFFMPEGAugmenter):
    def __init__(
        self, second_video_path: str, use_second_audio: bool, orientation: str
    ):
        assert (
            type(use_second_audio) == bool
        ), "Expected a boolean value for use_second_audio"
        assert orientation in [
            "hstack",
            "vstack",
        ], "Expected orientation to be either 'hstack' or 'vstack'"

        self.second_video_path = pathmgr.get_local_path(second_video_path)
        self.use_second_audio = use_second_audio
        self.orientation = orientation

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Stacks two videos together

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])

        second_video = ffmpeg.input(self.second_video_path)
        scaled_second_video = second_video.filter(
            "scale", **{"width": video_info["width"], "height": video_info["height"]}
        )
        stack_video = ffmpeg.filter([in_stream, scaled_second_video], self.orientation)

        return (
            (
                ffmpeg.concat(stack_video, second_video.audio, v=1, a=1)
                if self.use_second_audio
                else stack_video
            ),
            {},
        )
