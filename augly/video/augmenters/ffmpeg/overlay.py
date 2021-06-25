#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

import ffmpeg  # @manual
from augly.utils import is_image_file, is_video_file, pathmgr
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByOverlay(BaseFFMPEGAugmenter):
    def __init__(
        self,
        overlay_path: str,
        x_factor: float,
        y_factor: float,
        use_overlay_audio: bool,
    ):
        assert is_image_file(overlay_path) or is_video_file(
            overlay_path
        ), "Overlaid media type not supported: please overlay either an image or video"
        assert 0 <= x_factor <= 1, "x_factor must be a value in the range [0, 1]"
        assert 0 <= y_factor <= 1, "y_factor must be a value in the range [0, 1]"
        assert (
            type(use_overlay_audio) == bool
        ), "Expected a boolean value for use_overlay_audio"

        self.overlay_path = pathmgr.get_local_path(overlay_path)
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.use_overlay_audio = use_overlay_audio and is_video_file(overlay_path)

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Overlays media onto the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])

        new_width = video_info["width"] * self.x_factor
        new_height = video_info["height"] * self.y_factor

        overlay = ffmpeg.input(self.overlay_path)
        overlayed_video = ffmpeg.overlay(in_stream, overlay, x=new_width, y=new_height)

        return (
            (
                ffmpeg.concat(overlayed_video, overlay.audio, v=1, a=1)
                if self.use_overlay_audio
                else overlayed_video
            ),
            {},
        )
