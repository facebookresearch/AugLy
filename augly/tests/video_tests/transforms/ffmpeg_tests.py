#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
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

    def test_AddNoise(self):
        self.evaluate_class(vidaugs.AddNoise(level=10), fname="add_noise")

    def test_AudioSwap(self):
        self.evaluate_class(
            vidaugs.AudioSwap(audio_path=self.config.input_audio_file),
            fname="audio_swap",
        )

    def test_Blur(self):
        self.evaluate_class(vidaugs.Blur(sigma=3), fname="blur")

    def test_Brightness(self):
        self.evaluate_class(vidaugs.Brightness(), fname="brightness")

    def test_ChangeAspectRatio(self):
        self.evaluate_class(vidaugs.ChangeAspectRatio(), fname="change_aspect_ratio")

    def test_ChangeVideoSpeed(self):
        self.evaluate_class(
            vidaugs.ChangeVideoSpeed(factor=2.0), fname="change_video_speed"
        )

    def test_ColorJitter(self):
        self.evaluate_class(
            vidaugs.ColorJitter(
                brightness_factor=0.15, contrast_factor=1.3, saturation_factor=2.0
            ),
            fname="color_jitter",
        )

    def test_Concat(self):
        video_path, _ = VideoAugConfig(input_file_index=0).get_input_path()
        second_path, _ = VideoAugConfig(input_file_index=1).get_input_path()
        self.evaluate_class(
            vidaugs.Concat(other_video_paths=[second_path]),
            diff_video_input=True,
            video_path=video_path,
            fname="concat",
        )

    def test_Contrast(self):
        self.evaluate_class(vidaugs.Contrast(level=1.3), fname="contrast")

    def test_Crop(self):
        self.evaluate_class(vidaugs.Crop(), fname="crop")

    def test_EncodingQuality(self):
        self.evaluate_class(vidaugs.EncodingQuality(quality=37), fname="encoding_quality")

    def test_FPS(self):
        self.evaluate_class(vidaugs.FPS(), fname="fps")

    def test_Grayscale(self):
        self.evaluate_class(vidaugs.Grayscale(), fname="grayscale")

    def test_HFlip(self):
        self.evaluate_class(vidaugs.HFlip(), fname="hflip")

    def test_HStack(self):
        second_video_path, _ = VideoAugConfig(input_file_index=1).get_input_path()
        self.evaluate_class(
            vidaugs.HStack(second_video_path=second_video_path), fname="hstack"
        )

    def test_Loop(self):
        self.evaluate_class(vidaugs.Loop(num_loops=1), fname="loop")

    def test_Overlay(self):
        overlay_path, _ = VideoAugConfig(input_file_index=1).get_input_path()
        self.evaluate_class(vidaugs.Overlay(overlay_path=overlay_path), fname="overlay")

    def test_Pad(self):
        self.evaluate_class(vidaugs.Pad(), fname="pad")

    def test_RandomAspectRatio(self):
        self.evaluate_class(vidaugs.RandomAspectRatio(), fname="RandomAspectRatio", seed=1)

    def test_RandomBlur(self):
        self.evaluate_class(vidaugs.RandomBlur(), fname="RandomBlur", seed=1)

    def test_RandomBrightness(self):
        self.evaluate_class(vidaugs.RandomBrightness(), fname="RandomBrightness", seed=1)

    def test_RandomContrast(self):
        self.evaluate_class(vidaugs.RandomContrast(), fname="RandomContrast", seed=1)

    def test_RandomEncodingQuality(self):
        self.evaluate_class(
            vidaugs.RandomEncodingQuality(), fname="RandomEncodingQuality", seed=1
        )

    def test_RandomFPS(self):
        self.evaluate_class(vidaugs.RandomFPS(), fname="RandomFPS", seed=1)

    def test_RandomNoise(self):
        self.evaluate_class(vidaugs.RandomNoise(), fname="RandomNoise", seed=1)

    def test_RandomRotation(self):
        self.evaluate_class(vidaugs.RandomRotation(), fname="RandomRotation", seed=1)

    def test_RandomVideoSpeed(self):
        self.evaluate_class(vidaugs.RandomVideoSpeed(), fname="RandomVideoSpeed", seed=1)

    def test_RemoveAudio(self):
        self.evaluate_class(vidaugs.RemoveAudio(), fname="remove_audio")

    def test_Resize(self):
        self.evaluate_class(vidaugs.Resize(height=300, width=300), fname="resize")

    def test_Rotate(self):
        self.evaluate_class(vidaugs.Rotate(), fname="rotate")

    def test_Scale(self):
        self.evaluate_class(vidaugs.Scale(), fname="scale")

    def test_TimeCrop(self):
        self.evaluate_class(vidaugs.TimeCrop(duration_factor=0.5), fname="time_crop")

    def test_TimeDecimate(self):
        self.evaluate_class(vidaugs.TimeDecimate(), fname="time_decimate")

    def test_Trim(self):
        self.evaluate_class(vidaugs.Trim(end=5), fname="trim")

    def test_VFlip(self):
        self.evaluate_class(vidaugs.VFlip(), fname="vflip")

    def test_VStack(self):
        second_video_path, _ = VideoAugConfig(input_file_index=1).get_input_path()
        self.evaluate_class(
            vidaugs.VStack(second_video_path=second_video_path), fname="vstack"
        )


if __name__ == "__main__":
    unittest.main()
