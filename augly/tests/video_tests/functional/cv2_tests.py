#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
import unittest

import augly.video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.utils import pathmgr, FONT_PATH
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


@unittest.skipUnless(*get_conditional_for_skipping_video_tests())
class CV2VideoUnitTest(BaseVideoUnitTest):
    def test_overlay_dots(self):
        random.seed(1)
        self.evaluate_function(vidaugs.overlay_dots)

    def test_overlay_shapes(self):
        random.seed(1)
        self.evaluate_function(vidaugs.overlay_shapes)

    def test_overlay_text(self):
        random.seed(1)
        self.evaluate_function(
            vidaugs.overlay_text,
            topleft=(0, 0),
            bottomright=(0.5, 0.25),
        )

    def test_overlay_text_with_pil_font(self):
        font_local_path = pathmgr.get_local_path(FONT_PATH)
        random.seed(1)
        self.evaluate_function(
            vidaugs.overlay_text,
            ref_filename="overlay_text_with_pil_font",
            fonts=[(font_local_path, f"{FONT_PATH[:-3]}pkl")],
            topleft=(0, 0),
            bottomright=(0.5, 0.25),
        )


if __name__ == "__main__":
    unittest.main()
