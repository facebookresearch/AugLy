#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Tuple

import ffmpeg  # @manual
from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_audio_info, get_video_info
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByAudioSwap(BaseFFMPEGAugmenter):
    def __init__(self, audio_path: str, offset: float):
        assert offset >= 0, "Offset cannot be a negative number"

        self.audio_path = pathmgr.get_local_path(audio_path)
        self.offset = offset

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Swaps the audio of a video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        audio_info = get_audio_info(self.audio_path)
        video_info = get_video_info(kwargs["video_path"])

        audio_duration = float(audio_info["duration"])
        audio_sample_rate = float(audio_info["sample_rate"])

        start = self.offset
        end = start + float(video_info["duration"])

        audio = ffmpeg.input(self.audio_path).audio

        if end > audio_duration:
            pad_len = (end - audio_duration) * audio_sample_rate
            audio = audio.filter_("apad", pad_len=pad_len)

        audio = audio.filter_("atrim", start=start, end=end).filter_(
            "asetpts", "PTS-STARTPTS"
        )

        return ffmpeg.concat(in_stream.video, audio, v=1, a=1, n=1), {}
