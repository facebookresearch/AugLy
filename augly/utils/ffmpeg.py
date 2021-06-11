#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Tuple

FFMPEG_BIN_DIR = "/usr/local/bin"
FFMPEG_PATH = os.path.join(FFMPEG_BIN_DIR, "ffmpeg")
FFPROBE_PATH = os.path.join(FFMPEG_BIN_DIR, "ffprobe")


def get_conditional_for_skipping_video_tests() -> Tuple[bool, str]:
    return (True, "We currently do not need to skip any video tests")
