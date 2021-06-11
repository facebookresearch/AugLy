#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import augly.video as vidaugs
from augly.tests.video_tests.base_unit_test import BaseVideoUnitTest
from augly.utils.ffmpeg import get_conditional_for_skipping_video_tests


@unittest.skipUnless(*get_conditional_for_skipping_video_tests())
class FFMPEGVideoUnitTest(BaseVideoUnitTest):
    def test_add_noise(self):
        self.evaluate_function(vidaugs.add_noise, level=10)

    def test_audio_swap(self):
        self.evaluate_function(
            vidaugs.audio_swap, audio_path=self.config.input_audio_file
        )

    def test_blur(self):
        self.evaluate_function(vidaugs.blur, sigma=3)

    def test_brightness(self):
        self.evaluate_function(vidaugs.brightness)

    def test_change_aspect_ratio(self):
        self.evaluate_function(vidaugs.change_aspect_ratio)

    def test_change_video_speed(self):
        self.evaluate_function(vidaugs.change_video_speed, factor=2.0)

    def test_color_jitter(self):
        self.evaluate_function(
            vidaugs.color_jitter,
            brightness_factor=0.15,
            contrast_factor=1.3,
            saturation_factor=2.0,
        )

    def test_concat(self):
        _, second_path, _ = self.download_video(1)
        self.evaluate_function(
            vidaugs.concat,
            video_paths=[self.local_vid_path, second_path],
            diff_video_input=True,
        )

    def test_contrast(self):
        self.evaluate_function(vidaugs.contrast, level=1.3)

    def test_crop(self):
        self.evaluate_function(vidaugs.crop)

    def test_encoding_quality(self):
        self.evaluate_function(vidaugs.encoding_quality, quality=37)

    def test_fps(self):
        self.evaluate_function(vidaugs.fps)

    def test_grayscale(self):
        self.evaluate_function(vidaugs.grayscale)

    def test_hflip(self):
        self.evaluate_function(vidaugs.hflip)

    def test_hstack(self):
        _, second_video_path, _ = self.download_video(1)
        self.evaluate_function(vidaugs.hstack, second_video_path=second_video_path)

    def test_loop(self):
        self.evaluate_function(vidaugs.loop, num_loops=1)

    def test_overlay(self):
        _, overlay_path, _ = self.download_video(1)
        self.evaluate_function(vidaugs.overlay, overlay_path=overlay_path)

    def test_pad(self):
        self.evaluate_function(vidaugs.pad)

    def test_remove_audio(self):
        self.evaluate_function(vidaugs.remove_audio)

    def test_resize(self):
        self.evaluate_function(vidaugs.resize, height=300, width=300)

    def test_rotate(self):
        self.evaluate_function(vidaugs.rotate)

    def test_scale(self):
        self.evaluate_function(vidaugs.scale)

    def test_shift(self):
        self.evaluate_function(vidaugs.shift, x_factor=0.25, y_factor=0.25)

    def test_time_crop(self):
        self.evaluate_function(vidaugs.time_crop, duration_factor=0.5)

    def test_time_decimate(self):
        self.evaluate_function(vidaugs.time_decimate)

    def test_trim(self):
        self.evaluate_function(vidaugs.trim, end=5)

    def test_vflip(self):
        self.evaluate_function(vidaugs.vflip)

    def test_vstack(self):
        _, second_video_path, _ = self.download_video(1)
        self.evaluate_function(vidaugs.vstack, second_video_path=second_video_path)


if __name__ == "__main__":
    unittest.main()
