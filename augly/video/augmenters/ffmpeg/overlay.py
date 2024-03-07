#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.utils import is_image_file, is_video_file, pathmgr
from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info, has_audio_stream


class VideoAugmenterByOverlay(BaseVidgearFFMPEGAugmenter):
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

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Overlays media onto the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        video_info = get_video_info(video_path)

        new_width = video_info["width"] * self.x_factor
        new_height = video_info["height"] * self.y_factor

        process_audio = has_audio_stream(video_path)

        ret = [
            *self.input_fmt(video_path),
            "-i",
            self.overlay_path,
            "-filter_complex",
            f"[0:v][1:v] overlay={new_width}:{new_height}",
        ]

        if process_audio:
            ret += [
                "-map",
                f"{int(self.use_overlay_audio)}:a:0",
            ]

        ret += self.output_fmt(output_path)

        return ret
