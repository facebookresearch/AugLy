#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
import random
import unittest

import augly.video as vidaugs
from augly.tests.base_configs import VideoAugConfig
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

    def test_ApplyLambda(self):
        self.evaluate_class(vidaugs.ApplyLambda(), fname="apply_lambda")

    def test_InsertInBackground(self):
        self.evaluate_class(
            vidaugs.InsertInBackground(offset_factor=0.25),
            fname="insert_in_background",
            metadata_exclude_keys=["dst_duration", "dst_fps", "intensity"],
        )

    def test_Compose(self):
        random.seed(1)
        self.evaluate_class(
            vidaugs.Compose(
                [
                    vidaugs.VFlip(),
                    vidaugs.Brightness(),
                    vidaugs.OneOf([vidaugs.Grayscale(), vidaugs.ApplyLambda()]),
                ]
            ),
            fname="compose",
        )

    def test_OverlayEmoji(self):
        self.evaluate_class(vidaugs.OverlayEmoji(), fname="overlay_emoji")

    def test_OverlayOntoBackgroundVideo(self):
        self.evaluate_class(
            vidaugs.OverlayOntoBackgroundVideo(
                background_path=VideoAugConfig(input_file_index=1).get_input_path()[0],
            ),
            fname="overlay_onto_background_video",
        )

    def test_Pixelization(self):
        self.evaluate_class(vidaugs.Pixelization(ratio=0.1), fname="pixelization")

    def test_RandomEmojiOverlay(self):
        random.seed(1)
        self.evaluate_class(
            vidaugs.RandomEmojiOverlay(),
            fname="RandomEmojiOverlay",
            metadata_exclude_keys=["emoji_path"],
        )

    def test_RandomPixelization(self):
        random.seed(1)
        self.evaluate_class(
            vidaugs.RandomPixelization(max_ratio=0.2), fname="RandomPixelization"
        )

    def test_ReplaceWithBackground(self):
        self.evaluate_class(
            vidaugs.ReplaceWithBackground(
                source_offset=0.1,
                background_offset=0,
                source_percentage=0.7,
            ),
            fname="replace_with_background",
        )

    def test_ReplaceWithColorFrames(self):
        self.evaluate_class(
            vidaugs.ReplaceWithColorFrames(), fname="replace_with_color_frames"
        )

    def test_Shift(self):
        self.evaluate_class(vidaugs.Shift(x_factor=0.25, y_factor=0.25), fname="shift")


if __name__ == "__main__":
    unittest.main()
