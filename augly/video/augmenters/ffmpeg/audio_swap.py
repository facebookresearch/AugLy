#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_audio_info, get_video_info


class VideoAugmenterByAudioSwap(BaseVidgearFFMPEGAugmenter):
    def __init__(self, audio_path: str, offset: float):
        assert offset >= 0, "Offset cannot be a negative number"

        self.audio_path = pathmgr.get_local_path(audio_path)
        self.offset = offset

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Swaps the audio of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        audio_info = get_audio_info(self.audio_path)
        video_info = get_video_info(video_path)

        audio_duration = float(audio_info["duration"])
        audio_sample_rate = float(audio_info["sample_rate"])

        start = self.offset
        end = start + float(video_info["duration"])

        audio_filters = f"atrim={start}:{end}," + "asetpts=PTS-STARTPTS"

        if end > audio_duration:
            pad_len = (end - audio_duration) * audio_sample_rate
            audio_filters += f",apad=pad_len={pad_len}"

        return [
            *self.input_fmt(video_path),
            "-i",
            self.audio_path,
            "-c:v",
            "copy",
            "-af",
            audio_filters,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            output_path,
        ]
