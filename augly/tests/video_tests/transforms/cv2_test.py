#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

from augly import video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.utils import VIDEO_METADATA_PATH


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
