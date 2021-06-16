#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
from distutils import spawn
from typing import Tuple

FFMPEG_PATH = os.environ.get("AUGLY_FFMPEG_PATH", None)
FFPROBE_PATH = os.environ.get("AUGLY_FFPROBE_PATH", None)

if FFMPEG_PATH is None:
    FFMPEG_PATH = spawn.find_executable('ffmpeg')

if FFPROBE_PATH is None:
    FFPROBE_PATH = spawn.find_executable('ffprobe')

ffmpeg_paths_error = (
    "Please install 'ffmpeg' or set the {} & {} environment variables "
    "before running AugLy Video"
)

assert (
    FFMPEG_PATH is not None and FFPROBE_PATH is not None
), ffmpeg_paths_error.format("AUGLY_FFMPEG_PATH", "AUGLY_FFPROBE_PATH")


def get_conditional_for_skipping_video_tests() -> Tuple[bool, str]:
    return (True, "We currently do not need to skip any video tests")
