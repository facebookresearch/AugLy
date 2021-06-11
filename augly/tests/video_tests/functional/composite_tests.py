#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import augly.video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.tests.base_configs import VideoAugConfig
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


@unittest.skipUnless(*get_conditional_for_skipping_video_tests())
class CompositeVideoUnitTest(BaseVideoUnitTest):
    def test_apply_lambda(self):
        self.evaluate_function(vidaugs.apply_lambda)

    def test_insert_in_background(self):
        self.evaluate_function(vidaugs.insert_in_background, offset_factor=0.25)

    def test_overlay_emoji(self):
        self.evaluate_function(vidaugs.overlay_emoji)

    def test_overlay_onto_background_video(self):
        self.evaluate_function(
            vidaugs.overlay_onto_background_video,
            background_path=VideoAugConfig(input_file_index=1).get_input_path()[0],
        )

    def test_pixelization(self):
        self.evaluate_function(vidaugs.pixelization, ratio=0.1)

    def test_replace_with_color_frames(self):
        self.evaluate_function(vidaugs.replace_with_color_frames)

    def test_shift(self):
        self.evaluate_function(vidaugs.shift, x_factor=0.25, y_factor=0.25)


if __name__ == "__main__":
    unittest.main()
