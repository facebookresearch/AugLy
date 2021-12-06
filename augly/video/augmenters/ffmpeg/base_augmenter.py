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
import shutil
import tempfile
from abc import ABC, abstractmethod
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import ffmpeg  # @manual
from augly.utils.ffmpeg import FFMPEG_PATH
from augly.video.helpers import has_audio_stream, validate_input_and_output_paths
from ffmpeg.nodes import FilterableStream
from vidgear.gears import WriteGear


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


class BaseVidgearFFMPEGAugmenter(ABC):
    def add_augmenter(
        self, video_path: str, output_path: Optional[str] = None, **kwargs
    ) -> None:
        """
        Applies the specific augmentation to the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param kwargs: parameters for specific augmenters
        """
        video_path, output_path = validate_input_and_output_paths(
            video_path, output_path
        )
        with tempfile.NamedTemporaryFile(
<<<<<<< HEAD
            suffix=os.path.splitext(video_path)[1]
=======
            suffix=video_path[video_path.index(".") :]
>>>>>>> 888e5d6... fix nits in documentation and remove unnecessary stuff
        ) as tmpfile:
            if video_path == output_path:
                shutil.copyfile(video_path, tmpfile.name)
                video_path = tmpfile.name
<<<<<<< HEAD
            writer = WriteGear(output_filename=output_path, logging=True)
=======
            writer = WriteGear(output_filename=video_path, logging=True)
>>>>>>> 888e5d6... fix nits in documentation and remove unnecessary stuff
            writer.execute_ffmpeg_cmd(self.get_command(video_path, output_path))
            writer.close()

    @abstractmethod
    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Constructs the FFMPEG command for VidGear

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
<<<<<<< HEAD
<<<<<<< HEAD

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
=======
            If not passed in, the original video file will be overwritten
=======
>>>>>>> 888e5d6... fix nits in documentation and remove unnecessary stuff

        @returns: a list of strings of the FFMPEG command if it were to be written
            in a command line
>>>>>>> c95167c... added base class vidgear
        """
        raise NotImplementedError("Implement get_command method")
