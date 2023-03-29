#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from augly import audio as audaugs
from augly.tests.audio_tests.base_unit_test import BaseAudioUnitTest


class FunctionalAudioUnitTest(BaseAudioUnitTest):
    def test_add_background_noise(self):
        self.evaluate_function(
            audaugs.add_background_noise, background_audio=None, snr_level_db=10.0
        )

    def test_apply_lambda(self):
        self.evaluate_function(audaugs.apply_lambda)

    def test_change_volume(self):
        self.evaluate_function(audaugs.change_volume, volume_db=10.0)

    def test_clicks(self):
        self.evaluate_function(audaugs.clicks, seconds_between_clicks=0.5)

    def test_clip(self):
        self.evaluate_function(audaugs.clip, offset_factor=0.25, duration_factor=0.5)

    def test_harmonic(self):
        self.evaluate_function(audaugs.harmonic, kernel_size=31, power=2.0, margin=1.0)

    def test_high_pass_filter(self):
        self.evaluate_function(audaugs.high_pass_filter, cutoff_hz=3000)

    def test_insert_in_background(self):
        self.evaluate_function(audaugs.insert_in_background, offset_factor=0.3)

    def test_invert_channels(self):
        self.evaluate_function(audaugs.invert_channels)

    def test_loop(self):
        self.evaluate_function(audaugs.loop, n=1)

    def test_low_pass_filter(self):
        self.evaluate_function(audaugs.low_pass_filter, cutoff_hz=500)

    def test_normalize(self):
        self.evaluate_function(audaugs.normalize)

    def test_peaking_equalizer(self):
        self.evaluate_function(audaugs.peaking_equalizer, gain_db=-20.0)

    def test_percussive(self):
        self.evaluate_function(
            audaugs.percussive, kernel_size=31, power=2.0, margin=1.0
        )

    def test_pitch_shift(self):
        self.evaluate_function(audaugs.pitch_shift, n_steps=4)

    def test_reverb(self):
        self.evaluate_function(audaugs.reverb, reverberance=100.0)

    def test_speed(self):
        self.evaluate_function(audaugs.speed, factor=3.0)

    def test_tempo(self):
        self.evaluate_function(audaugs.tempo, factor=2.0)

    def test_time_stretch(self):
        self.evaluate_function(audaugs.time_stretch, rate=1.5)

    def test_to_mono(self):
        self.evaluate_function(audaugs.to_mono)

    def test_fft_convolve(self):
        self.evaluate_function(audaugs.fft_convolve)


if __name__ == "__main__":
    unittest.main()
