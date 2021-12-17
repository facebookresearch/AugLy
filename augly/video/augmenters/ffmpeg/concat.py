#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from typing import List

from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info


class VideoAugmenterByConcat(BaseVidgearFFMPEGAugmenter):
    def __init__(self, video_paths: List[str], src_video_path_index: int):
        assert len(video_paths) > 0, "Please provide at least one input video"
        assert all(
            pathmgr.exists(video_path) for video_path in video_paths
        ), "Invalid video path(s) provided"

        self.video_paths = [
            pathmgr.get_local_path(video_path) for video_path in video_paths
        ]
        self.src_video_path_index = src_video_path_index

        video_info = get_video_info(self.video_paths[src_video_path_index])

        self.height = ceil(video_info["height"] / 2) * 2
        self.width = ceil(video_info["width"] / 2) * 2

        self.sample_aspect_ratio = video_info.get(
            "sample_aspect_ratio", self.width / self.height
        )

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Concatenates multiple videos together

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        inputs = [["-i", video] for video in self.video_paths]
        flat_inputs = [element for sublist in inputs for element in sublist]
        scale_and_sar, maps = "", ""
        for i in range(len(self.video_paths)):
            scale_and_sar += (
                f"[{i}:v]scale={self.width}:{self.height}[{i}v],[{i}v]setsar=ratio="
                f"{self.sample_aspect_ratio}[{i}vf];"
            )

        for i in range(len(self.video_paths)):
            maps += f"[{i}vf][{i}:a]"

        rest_command = f"concat=n={len(self.video_paths)}:v=1:a=1[v][a]"

        return [
            "-y",
            *flat_inputs,
            "-filter_complex",
            scale_and_sar + maps + rest_command,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-vsync",
            "2",
            *self.output_fmt(output_path),
        ]
