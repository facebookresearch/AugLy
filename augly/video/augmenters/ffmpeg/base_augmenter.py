#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from typing import List, Optional

from augly.video.helpers import validate_input_and_output_paths
from vidgear.gears import WriteGear


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
            suffix=os.path.splitext(video_path)[1]
        ) as tmpfile:
            if video_path == output_path:
                shutil.copyfile(video_path, tmpfile.name)
                video_path = tmpfile.name
            writer = WriteGear(output_filename=output_path, logging=True)
            writer.execute_ffmpeg_cmd(self.get_command(video_path, output_path))
            writer.close()

    @abstractmethod
    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Constructs the FFMPEG command for VidGear

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        raise NotImplementedError("Implement get_command method")
