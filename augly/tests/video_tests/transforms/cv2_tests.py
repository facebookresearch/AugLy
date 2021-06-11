#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
import unittest

import augly.video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.utils import VIDEO_METADATA_PATH
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


@unittest.skipUnless(*get_conditional_for_skipping_video_tests())
class TransformsVideoUnitTest(BaseVideoUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(VIDEO_METADATA_PATH, "r") as f:
            cls.metadata = json.load(f)

    def test_OverlayDots(self):
        self.evaluate_class(vidaugs.OverlayDots(), fname="overlay_dots", seed=1)

    def test_OverlayShapes(self):
        self.evaluate_class(vidaugs.OverlayShapes(), fname="overlay_shapes", seed=1)

    def test_OverlayText(self):
        self.evaluate_class(
            vidaugs.OverlayText(
                topleft=(0, 0),
                bottomright=(0.5, 0.25),
            ),
            fname="overlay_text",
            seed=1,
        )


if __name__ == "__main__":
    unittest.main()
