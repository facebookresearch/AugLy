#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Implementation of base class for FFMPEG-based video augmenters

- Method to override:
    - `add_augmenter(self, in_stream: FilterableStream, **kwargs)`:
      takes as input the FFMPEG video object and returns the output FFMPEG object
      with the augmentation applied along with a dictionary containing output
      arguments if needed.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import ffmpeg  # @manual
from ffmpeg.nodes import FilterableStream
from augly.utils.ffmpeg import FFMPEG_PATH
from augly.video.helpers import has_audio_stream


class BaseFFMPEGAugmenter(ABC):
    def augment(self, video_temp_dir: str, video_temp_path: str, **kwargs) -> str:
        """
        Augments a video (resolution change, etc.)

        @param video_temp_dir: local temp directory storing the video

        @param video_temp_path: local temp path of the video that needs augmentation

        @param kwargs: parameters for specific augmenters

        @returns: the path to the new video
        """
        output_path = os.path.join(video_temp_dir, "augmenter_final.mp4")
        in_stream = ffmpeg.input(video_temp_path)
        kwargs = {"video_path": video_temp_path, **kwargs}
        video, outputargs = self.add_augmenter(in_stream, **kwargs)
        video = video.filter(  # pyre-fixme[16]: `FilterableStream` has no attribute `filter`
            "pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"}
        )
        audio = in_stream.audio
        output = (
            ffmpeg.output(video, audio, output_path, **outputargs)
            if has_audio_stream(video_temp_path)
            else ffmpeg.output(video, output_path, **outputargs)
        )
        output.overwrite_output().run(cmd=FFMPEG_PATH)
        return output_path

    @abstractmethod
    def add_augmenter(
        self, in_stream: FilterableStream, **kwargs
    ) -> Tuple[FilterableStream, Dict]:
        """
        Applies the specific augmentation to the video

        @param in_stream: the FFMPEG object of the video

        @returns: a tuple containing the FFMPEG object with the augmentation
            applied and a dictionary with any output arguments as necessary
        """
        raise NotImplementedError("Implement add_augmenter method")
