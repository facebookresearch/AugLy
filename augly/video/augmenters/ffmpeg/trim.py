#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional, Tuple

import ffmpeg  # @manual
from augly.video.augmenters.ffmpeg import BaseFFMPEGAugmenter
from augly.video.helpers import get_video_info, has_audio_stream
from ffmpeg.nodes import FilterableStream


class VideoAugmenterByTrim(BaseFFMPEGAugmenter):
    def __init__(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        offset_factor: float = 0.0,
        duration_factor: float = 1.0,
    ):
        assert start is None or start >= 0, "Start cannot be a negative number"
        assert (
            end is None or (start is not None and end > start) or end > 0
        ), "End must be a value greater than start"
        assert (
            0.0 <= offset_factor <= 1.0
        ), "Offset factor must be a value in the range [0.0, 1.0]"
        assert (
            0.0 <= duration_factor <= 1.0
        ), "Duration factor must be a value in the range [0.0, 1.0]"

        if start is not None or end is not None:
            self.start = start
            self.end = end
            self.offset_factor = None
            self.duration_factor = None
        else:
            self.start = None
            self.end = None
            self.offset_factor = offset_factor
            self.duration_factor = duration_factor

    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Trims the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        video_info = get_video_info(kwargs["video_path"])
        duration = float(video_info["duration"])

        if self.start is None and self.end is None:
            self.start = self.offset_factor * duration
            duration = min(self.duration_factor * duration, duration - self.start)
            self.end = self.start + duration
        elif self.start is None:
            self.start = 0
        elif self.end is None:
            self.end = duration

        video = in_stream.video.trim(start=self.start, end=self.end).setpts(
            "PTS-STARTPTS"
        )

        if has_audio_stream(kwargs["video_path"]):
            audio = in_stream.audio.filter_(
                "atrim", start=self.start, end=self.end
            ).filter_("asetpts", "PTS-STARTPTS")

            return ffmpeg.concat(video, audio, v=1, a=1), {}

        return video, {}
