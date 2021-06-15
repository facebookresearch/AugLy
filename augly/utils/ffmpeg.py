#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Tuple

FFMPEG_PATH = os.environ.get("AUGLY_FFMPEG_PATH")
FFPROBE_PATH = os.environ.get("AUGLY_FFPROBE_PATH")

assert (
    FFMPEG_PATH is not None
), "Please set the AUGLY_FFMPEG_PATH environment variable before running AugLy Video"

assert (
    FFPROBE_PATH is not None
), "Please set the AUGLY_FFPROBE_PATH environment variable before running AugLy Video"


def get_conditional_for_skipping_video_tests() -> Tuple[bool, str]:
    return (True, "We currently do not need to skip any video tests")
