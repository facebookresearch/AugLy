#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os

MODULE_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# asset paths
ASSETS_BASE_DIR = os.path.join(MODULE_BASE_DIR, "assets")

AUDIO_ASSETS_DIR = os.path.join(ASSETS_BASE_DIR, "audio")
TEXT_DIR = os.path.join(ASSETS_BASE_DIR, "text")
EMOJI_DIR = os.path.join(ASSETS_BASE_DIR, "twemojis")
FONTS_DIR = os.path.join(ASSETS_BASE_DIR, "fonts")
IMG_MASK_DIR = os.path.join(ASSETS_BASE_DIR, "masks")
SCREENSHOT_TEMPLATES_DIR = os.path.join(ASSETS_BASE_DIR, "screenshot_templates")
TEMPLATE_PATH = os.path.join(SCREENSHOT_TEMPLATES_DIR, "web.png")

TEST_URI = os.path.join(ASSETS_BASE_DIR, "tests")

# test paths
METADATA_BASE_PATH = os.path.join(MODULE_BASE_DIR, "utils", "expected_output")
METADATA_FILENAME = "expected_metadata.json"

AUDIO_METADATA_PATH = os.path.join(METADATA_BASE_PATH, "audio_tests", METADATA_FILENAME)
IMAGE_METADATA_PATH = os.path.join(METADATA_BASE_PATH, "image_tests", METADATA_FILENAME)
TEXT_METADATA_PATH = os.path.join(METADATA_BASE_PATH, "text_tests", METADATA_FILENAME)
VIDEO_METADATA_PATH = os.path.join(METADATA_BASE_PATH, "video_tests", METADATA_FILENAME)
