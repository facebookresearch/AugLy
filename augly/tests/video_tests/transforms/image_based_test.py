#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

from aml.augly.tests.video_tests.base_unit_test import BaseVideoUnitTest

from augly import video as vidaugs
from augly.tests.base_configs import VideoAugConfig
from augly.utils import VIDEO_METADATA_PATH
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


class TransformsVideoUnitTest(BaseVideoUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(VIDEO_METADATA_PATH, "r") as f:
            cls.metadata = json.load(f)

    def test_BlendVideos(self):
        overlay_path, _ = VideoAugConfig(input_file_index=1).get_input_path()
        self.evaluate_class(
            vidaugs.BlendVideos(overlay_path=overlay_path, opacity=0.5),
            fname="blend_videos",
        )

    def test_MemeFormat(self):
        self.evaluate_class(vidaugs.MemeFormat(), fname="meme_format")

    def test_OverlayOntoScreenshot(self):
        self.evaluate_class(
            vidaugs.OverlayOntoScreenshot(),
            fname="overlay_onto_screenshot",
            metadata_exclude_keys=[
                "dst_height",
                "dst_width",
                "intensity",
                "template_filepath",
            ],
        )

    @unittest.skipUnless(*get_conditional_for_skipping_video_tests())
    def test_PerspectiveTransformAndShake(self):
        self.evaluate_class(
            vidaugs.PerspectiveTransformAndShake(shake_radius=20, seed=10),
            fname="perspective_transform_and_shake",
        )


if __name__ == "__main__":
    unittest.main()
