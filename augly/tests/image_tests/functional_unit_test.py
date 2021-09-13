#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import augly.image as imaugs
from augly.tests.image_tests.base_unit_test import BaseImageUnitTest
from augly.utils import EMOJI_PATH, IMG_MASK_PATH


class FunctionalImageUnitTest(BaseImageUnitTest):
    def test_apply_lambda(self):
        self.evaluate_function(imaugs.apply_lambda)

    def test_apply_pil_filter(self):
        self.evaluate_function(imaugs.apply_pil_filter)

    def test_blur(self):
        self.evaluate_function(imaugs.blur)

    def test_brightness(self):
        self.evaluate_function(imaugs.brightness)

    def test_change_aspect_ratio(self):
        self.evaluate_function(imaugs.change_aspect_ratio)

    def test_clip_image_size(self):
        self.evaluate_function(imaugs.clip_image_size, max_resolution=1500000)

    def test_color_jitter(self):
        self.evaluate_function(imaugs.color_jitter)

    def test_contrast(self):
        self.evaluate_function(imaugs.contrast)

    def test_convert_color(self):
        self.evaluate_function(imaugs.convert_color, mode="L")

    def test_crop(self):
        self.evaluate_function(imaugs.crop)

    def test_encoding_quality(self):
        self.evaluate_function(imaugs.encoding_quality, quality=30)

    def test_grayscale(self):
        self.evaluate_function(imaugs.grayscale)

    def test_hflip(self):
        self.evaluate_function(imaugs.hflip)

    def test_masked_composite(self):
        self.evaluate_function(
            imaugs.masked_composite,
            mask=IMG_MASK_PATH,
            transform_function=imaugs.Brightness(factor=0.1),
        )

    @unittest.skip("Failing on some envs, will fix")
    def test_meme_format(self):
        self.evaluate_function(imaugs.meme_format)

    def test_opacity(self):
        self.evaluate_function(imaugs.opacity)

    def test_overlay_emoji(self):
        self.evaluate_function(imaugs.overlay_emoji)

    def test_overlay_image(self):
        self.evaluate_function(
            imaugs.overlay_image, overlay=EMOJI_PATH, overlay_size=0.15, y_pos=0.8
        )

    def test_overlay_onto_background_image(self):
        self.evaluate_function(
            imaugs.overlay_onto_background_image,
            background_image=EMOJI_PATH,
            overlay_size=0.5,
            scale_bg=True,
        )

    def test_overlay_onto_screenshot(self):
        self.evaluate_function(
            imaugs.overlay_onto_screenshot, resize_src_to_match_template=False
        )

    def test_overlay_stripes(self):
        self.evaluate_function(imaugs.overlay_stripes)

    @unittest.skip("Failing on some envs, will fix")
    def test_overlay_text(self):
        text_indices = [5, 3, 1, 2, 1000, 221]
        self.evaluate_function(imaugs.overlay_text, text=text_indices)

    def test_pad(self):
        self.evaluate_function(imaugs.pad)

    def test_pad_square(self):
        self.evaluate_function(imaugs.pad_square)

    def test_perspective_transform(self):
        self.evaluate_function(imaugs.perspective_transform, sigma=100.0)

    def test_pixelization(self):
        self.evaluate_function(imaugs.pixelization)

    def test_random_noise(self):
        self.evaluate_function(imaugs.random_noise)

    def test_resize(self):
        self.evaluate_function(imaugs.resize)

    def test_rotate(self):
        self.evaluate_function(imaugs.rotate)

    def test_saturation(self):
        self.evaluate_function(imaugs.saturation, factor=0.5)

    def test_scale(self):
        self.evaluate_function(imaugs.scale)

    def test_sharpen(self):
        self.evaluate_function(imaugs.sharpen, factor=2.0)

    def test_shuffle_pixels(self):
        self.evaluate_function(imaugs.shuffle_pixels, factor=0.5)

    def test_vflip(self):
        self.evaluate_function(imaugs.vflip)


if __name__ == "__main__":
    unittest.main()
