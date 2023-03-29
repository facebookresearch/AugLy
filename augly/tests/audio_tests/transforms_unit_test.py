#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import unittest

import numpy as np
from augly import audio as audaugs
from augly.tests.audio_tests.base_unit_test import BaseAudioUnitTest
from augly.utils import AUDIO_METADATA_PATH


class TransformsAudioUnitTest(BaseAudioUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(AUDIO_METADATA_PATH, "r") as f:
            cls.metadata = json.load(f)

    def test_AddBackgroundNoise(self):
        random_generator = np.random.default_rng(1)
        self.evaluate_class(
            audaugs.AddBackgroundNoise(
                background_audio=None, snr_level_db=10.0, seed=random_generator
            ),
            fname="add_background_noise2",
        )

    def test_ApplyLambda(self):
        self.evaluate_class(audaugs.ApplyLambda(), fname="apply_lambda")

    def test_ChangeVolume(self):
        self.evaluate_class(audaugs.ChangeVolume(volume_db=10.0), fname="change_volume")

    def test_Clicks(self):
        self.evaluate_class(audaugs.Clicks(seconds_between_clicks=0.5), fname="clicks")

    def test_Clip(self):
        self.evaluate_class(
            audaugs.Clip(offset_factor=0.25, duration_factor=0.5), fname="clip"
        )

    def test_Compose(self):
        self.evaluate_class(
            audaugs.Compose(
                [
                    audaugs.Clip(duration_factor=0.25),
                    audaugs.ChangeVolume(volume_db=10.0),
                ]
            ),
            fname="compose",
        )

    def test_Compose2(self):
        random_generator = np.random.default_rng(1)
        self.evaluate_class(
            audaugs.Compose(
                [
                    audaugs.InsertInBackground(
                        offset_factor=0.2, background_audio=None, seed=random_generator
                    ),
                    audaugs.Clip(offset_factor=0.5, duration_factor=0.25),
                    audaugs.Speed(factor=4.0),
                ]
            ),
            fname="compose2",
        )

    def test_Harmonic(self):
        self.evaluate_class(
            audaugs.Harmonic(kernel_size=31, power=2.0, margin=1.0), fname="harmonic"
        )

    def test_HighPassFilter(self):
        self.evaluate_class(
            audaugs.HighPassFilter(cutoff_hz=3000), fname="high_pass_filter"
        )

    def test_InsertInBackground(self):
        random_generator = np.random.default_rng(1)
        self.evaluate_class(
            audaugs.InsertInBackground(offset_factor=0.3, seed=random_generator),
            fname="insert_in_background2",
        )

    def test_InvertChannels(self):
        self.evaluate_class(audaugs.InvertChannels(), fname="invert_channels")

    def Loop(self):
        self.evaluate_class(audaugs.Loop(n=1), fname="loop")

    def test_LowPassFilter(self):
        self.evaluate_class(
            audaugs.LowPassFilter(cutoff_hz=500), fname="low_pass_filter"
        )

    def test_Normalize(self):
        self.evaluate_class(audaugs.Normalize(), fname="normalize")

    def test_OneOf(self):
        random.seed(1)
        self.evaluate_class(
            audaugs.OneOf(
                [audaugs.PitchShift(n_steps=4), audaugs.TimeStretch(rate=1.5)]
            ),
            fname="time_stretch",
        )

    def test_PeakingEqualizer(self):
        self.evaluate_class(
            audaugs.PeakingEqualizer(gain_db=-20.0), fname="peaking_equalizer"
        )

    def test_Percussive(self):
        self.evaluate_class(
            audaugs.Percussive(kernel_size=31, power=2.0, margin=1.0),
            fname="percussive",
        )

    def test_PitchShift(self):
        self.evaluate_class(audaugs.PitchShift(n_steps=4), fname="pitch_shift")

    def test_Reverb(self):
        self.evaluate_class(audaugs.Reverb(reverberance=100.0), fname="reverb")

    def test_Speed(self):
        self.evaluate_class(audaugs.Speed(factor=3.0), fname="speed")

    def test_Tempo(self):
        self.evaluate_class(audaugs.Tempo(factor=2.0), fname="tempo")

    def test_TimeStretch(self):
        self.evaluate_class(audaugs.TimeStretch(rate=1.5), fname="time_stretch")

    def test_ToMono(self):
        self.evaluate_class(audaugs.ToMono(), fname="to_mono")

    def test_FFTConvolve(self):
        np.random.seed(1)
        self.evaluate_class(audaugs.FFTConvolve(normalize=True), fname="fft_convolve")


if __name__ == "__main__":
    unittest.main()
