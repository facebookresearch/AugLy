#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from math import ceil
from typing import Dict, List, Tuple

import ffmpeg  # @manual
from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByConcat(BaseFFMPEGAugmenter):
    def __init__(self, video_paths: List[str], src_video_path_index: int):
        assert len(video_paths) > 0, "Please provide at least one input video"
        assert all(
            pathmgr.exists(video_path) for video_path in video_paths
        ), "Invalid video path(s) provided"

        video_paths = [pathmgr.get_local_path(video_path) for video_path in video_paths]
        self.videos = [ffmpeg.input(video_path) for video_path in video_paths]

        video_info = get_video_info(video_paths[src_video_path_index])

        self.height = ceil(video_info["height"] / 2) * 2
        self.width = ceil(video_info["width"] / 2) * 2

        self.sample_aspect_ratio = video_info.get(
            "sample_aspect_ratio", self.width / self.height
        )

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Concatenates multiple videos together

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        streams = []

        for stream in self.videos:
            streams.extend(
                (
                    stream.video.filter(
                        "scale", **{"width": self.width, "height": self.height}
                    ).filter("setsar", **{"ratio": self.sample_aspect_ratio}),
                    stream.audio,
                )
            )

        return ffmpeg.concat(*streams, v=1, a=1, n=len(streams) // 2), {"vsync": 2}
