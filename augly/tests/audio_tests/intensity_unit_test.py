#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from augly import audio as audaugs


class IntensityAudioUnitTest(unittest.TestCase):
    def test_add_background_noise_intensity(self):
        intensity = audaugs.add_background_noise_intensity(
            metadata={}, snr_level_db=10.0
        )
        self.assertAlmostEqual(intensity, 90.90909091)

    def test_apply_lambda_intensity(self):
        intensity = audaugs.apply_lambda_intensity(
            metadata={}, aug_function=lambda x, y: (x, y)
        )
        self.assertAlmostEqual(intensity, 100.0)

    def test_change_volume_intensity(self):
        intensity = audaugs.change_volume_intensity(metadata={}, volume_db=10.0)
        self.assertAlmostEqual(intensity, 9.090909091)

    def test_clicks_intensity(self):
        intensity = audaugs.clicks_intensity(
            metadata={}, seconds_between_clicks=0.5, snr_level_db=1.0
        )
        self.assertAlmostEqual(intensity, 98.26515152)

    def test_clip_intensity(self):
        intensity = audaugs.clip_intensity(metadata={}, duration_factor=0.75)
        self.assertAlmostEqual(intensity, 25.0)

    def test_harmonic_intensity(self):
        intensity = audaugs.harmonic_intensity(metadata={})
        self.assertAlmostEqual(intensity, 100.0)

    def test_high_pass_filter_intensity(self):
        intensity = audaugs.high_pass_filter_intensity(metadata={}, cutoff_hz=3000.0)
        self.assertAlmostEqual(intensity, 15.0)

    def test_insert_in_background_intensity(self):
        intensity = audaugs.insert_in_background_intensity(
            metadata={"src_duration": 10.0, "dst_duration": 15.0}
        )
        self.assertAlmostEqual(intensity, 33.3333333)

    def test_invert_channels_intensity(self):
        intensity = audaugs.invert_channels_intensity(metadata={"src_num_channels": 2})
        self.assertAlmostEqual(intensity, 100.0)

    def test_loop_intensity(self):
        intensity = audaugs.loop_intensity(metadata={}, n=1)
        self.assertAlmostEqual(intensity, 1.0)

    def test_low_pass_filter_intensity(self):
        intensity = audaugs.low_pass_filter_intensity(metadata={}, cutoff_hz=500.0)
        self.assertAlmostEqual(intensity, 97.5)

    def test_normalize_intensity(self):
        intensity = audaugs.normalize_intensity(metadata={}, norm=np.inf)
        self.assertAlmostEqual(intensity, 100.0)

    def test_peaking_equalizer_intensity(self):
        intensity = audaugs.peaking_equalizer_intensity(q=1.0, gain_db=-20.0)
        self.assertAlmostEqual(intensity, 17.786561264822133)

    def test_percussive_intensity(self):
        intensity = audaugs.percussive_intensity(metadata={})
        self.assertAlmostEqual(intensity, 100.0)

    def test_pitch_shift_intensity(self):
        intensity = audaugs.pitch_shift_intensity(metadata={}, n_steps=2.0)
        self.assertAlmostEqual(intensity, 2.380952381)

    def test_reverb_intensity(self):
        intensity = audaugs.reverb_intensity(
            metadata={}, reverberance=75.0, wet_only=False, room_scale=100.0
        )
        self.assertAlmostEqual(intensity, 75.0)

    def test_speed_intensity(self):
        intensity = audaugs.speed_intensity(metadata={}, factor=2.0)
        self.assertAlmostEqual(intensity, 20.0)

    def test_tempo_intensity(self):
        intensity = audaugs.tempo_intensity(metadata={}, factor=0.5)
        self.assertAlmostEqual(intensity, 20.0)

    def test_time_stretch_intensity(self):
        intensity = audaugs.time_stretch_intensity(metadata={}, factor=1.5)
        self.assertAlmostEqual(intensity, 15.0)

    def test_to_mono_intensity(self):
        intensity = audaugs.to_mono_intensity(metadata={"src_num_channels": 1})
        self.assertAlmostEqual(intensity, 0.0)
