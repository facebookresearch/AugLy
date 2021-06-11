#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import augly.video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


@unittest.skipUnless(*get_conditional_for_skipping_video_tests())
class ImageBasedVideoUnitTest(BaseVideoUnitTest):
    def test_blend_videos(self):
        _, overlay_path, _ = self.download_video(1)
        self.evaluate_function(
            vidaugs.blend_videos, overlay_path=overlay_path, opacity=0.5
        )

    def test_meme_format(self):
        self.evaluate_function(vidaugs.meme_format)

    def test_overlay_onto_screenshot(self):
        self.evaluate_function(vidaugs.overlay_onto_screenshot)

    def test_perspective_transform_and_shake(self):
        self.evaluate_function(
            vidaugs.perspective_transform_and_shake, shake_radius=20, seed=10
        )


if __name__ == "__main__":
    unittest.main()
