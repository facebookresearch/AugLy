#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from augly import image as imaugs
from augly.tests.image_tests.base_unit_test import BaseImageUnitTest
from augly.utils import EMOJI_PATH, IMG_MASK_PATH
from PIL import Image


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

    def test_collage(self):
        img = self.img.copy()
        result = imaugs.collage([img, img, img, img], n_columns=2)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.width, img.width * 2)
        self.assertEqual(result.height, img.height * 2)

    def test_hflip(self):
        self.evaluate_function(imaugs.hflip)

    def test_hstack(self):
        img = self.img.copy()
        result = imaugs.hstack([img, img])
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.width, img.width * 2)
        self.assertEqual(result.height, img.height)

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

    def test_overlay_random_text_with_background(self):
        result = imaugs.overlay_random_text_with_background(self.img, seed=42)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_with_custom_colors(self):
        result = imaugs.overlay_random_text_with_background(
            self.img,
            seed=42,
            text_color=(255, 255, 255),
            box_color=(0, 0, 0),
            box_opacity=0.8,
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_placement_top(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, placement="top"
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_placement_center(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, placement="center"
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_placement_random(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, placement="random"
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_rotation(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, text_rotation=15.0
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_bold(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, bold=True
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_stroke_width(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, stroke_width=3
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_gradient(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, box_bg_mode="gradient"
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_gradient_vertical(self):
        result = imaugs.overlay_random_text_with_background(
            self.img,
            seed=42,
            box_bg_mode="gradient",
            gradient_direction="vertical",
            gradient_end_color=(100, 100, 100),
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_num_overlays(self):
        result = imaugs.overlay_random_text_with_background(
            self.img, seed=42, num_overlays=2
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_all_features(self):
        result = imaugs.overlay_random_text_with_background(
            self.img,
            seed=42,
            placement="top",
            text_rotation=10.0,
            bold=True,
            box_bg_mode="gradient",
            num_overlays=2,
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_random_text_with_background_custom_phrases(self):
        result = imaugs.overlay_random_text_with_background(
            self.img,
            seed=42,
            phrases=["CUSTOM TEXT ONE", "CUSTOM TEXT TWO"],
        )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.img.size)

    def test_overlay_wrap_text(self):
        text = "Testing if the function can wrap this awesome text and not go out of bounds"
        self.evaluate_function(
            imaugs.overlay_wrap_text, text=text, font_size=0.2, random_seed=42
        )

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

    def test_ranking_numbers(self):
        ranking_dict = {1: "First place", 2: "Second place", 3: "Third place"}
        self.evaluate_function(imaugs.ranking_numbers, ranking_dict=ranking_dict)

    def test_resize(self):
        self.evaluate_function(imaugs.resize, resample=Image.BICUBIC)

    def test_rotate(self):
        self.evaluate_function(imaugs.rotate)

    def test_rotate_default_matches_no_expand(self):
        # Setup: rotate with default params and with explicit expand=False
        default_result = imaugs.rotate(self.img, degrees=30.0)
        no_expand_result = imaugs.rotate(self.img, degrees=30.0, expand=False)

        # Assert: both should produce identical output
        self.assertEqual(default_result.size, no_expand_result.size)
        self.assertEqual(
            list(default_result.getdata()), list(no_expand_result.getdata())
        )

    def test_rotate_expand(self):
        # Setup: rotate with and without expand
        degrees = 30.0
        no_expand_result = imaugs.rotate(self.img, degrees=degrees, expand=False)
        expand_result = imaugs.rotate(self.img, degrees=degrees, expand=True)

        # Assert: expanded image should be larger than center-cropped image
        self.assertGreater(
            expand_result.width * expand_result.height,
            no_expand_result.width * no_expand_result.height,
        )
        # Assert: expanded dimensions should match expected rotated bounding box
        import math

        rad = math.radians(degrees)
        expected_w = int(
            self.img.width * abs(math.cos(rad)) + self.img.height * abs(math.sin(rad))
        )
        expected_h = int(
            self.img.width * abs(math.sin(rad)) + self.img.height * abs(math.cos(rad))
        )
        self.assertAlmostEqual(expand_result.width, expected_w, delta=2)
        self.assertAlmostEqual(expand_result.height, expected_h, delta=2)

    def test_rotate_expand_with_fill_color(self):
        # Setup: rotate with expand and a white fill color
        degrees = 45.0
        fill_color = (255, 255, 255)
        result = imaugs.rotate(
            self.img,
            degrees=degrees,
            expand=True,
            fill_color=fill_color,
        )

        # Assert: result should be a valid PIL Image
        self.assertIsInstance(result, Image.Image)
        # Assert: corner pixel should be the fill color (corners are outside the
        # rotated image area)
        corner_pixel = result.getpixel((0, 0))
        self.assertEqual(corner_pixel[:3], fill_color)

    def test_rotate_expand_false_ignores_fill_color(self):
        # Setup: rotate without expand, passing fill_color which should be ignored
        degrees = 15.0
        result_no_fill = imaugs.rotate(self.img, degrees=degrees, expand=False)
        result_with_fill = imaugs.rotate(
            self.img,
            degrees=degrees,
            expand=False,
            fill_color=(255, 0, 0),
        )

        # Assert: both should produce identical output (fill_color has no effect
        # when expand=False because the image is center-cropped)
        self.assertEqual(result_no_fill.size, result_with_fill.size)
        self.assertEqual(
            list(result_no_fill.getdata()), list(result_with_fill.getdata())
        )

    def test_rotate_zero_degrees_expand(self):
        # Setup: rotate by 0 degrees with expand=True
        result = imaugs.rotate(self.img, degrees=0.0, expand=True)

        # Assert: should return an image with same dimensions as input
        self.assertEqual(result.size, self.img.size)

    def test_rotate_invalid_expand_type(self):
        # Assert: passing non-boolean expand should raise AssertionError
        with self.assertRaises(AssertionError):
            imaugs.rotate(self.img, degrees=15.0, expand="yes")

    def test_rotate_invalid_fill_color(self):
        # Assert: passing invalid fill_color should raise AssertionError
        with self.assertRaises(AssertionError):
            imaugs.rotate(self.img, degrees=15.0, expand=True, fill_color="red")
        with self.assertRaises(AssertionError):
            imaugs.rotate(self.img, degrees=15.0, expand=True, fill_color=(1, 2))

    def test_rotate_90_degrees_expand(self):
        # Setup: rotate by 90 degrees with expand=True
        result = imaugs.rotate(self.img, degrees=90.0, expand=True)

        # Assert: width and height should be swapped (for non-square images)
        self.assertEqual(result.width, self.img.height)
        self.assertEqual(result.height, self.img.width)

    def test_saturation(self):
        self.evaluate_function(imaugs.saturation, factor=0.5)

    def test_scale(self):
        self.evaluate_function(imaugs.scale)

    def test_sharpen(self):
        self.evaluate_function(imaugs.sharpen, factor=2.0)

    def test_shuffle_pixels(self):
        self.evaluate_function(imaugs.shuffle_pixels, factor=0.5)

    def test_skew(self):
        self.evaluate_function(imaugs.skew)

    def test_split_and_shuffle(self):
        self.evaluate_function(imaugs.split_and_shuffle, n_columns=3, n_rows=4)

    def test_vflip(self):
        self.evaluate_function(imaugs.vflip)

    def test_vstack(self):
        img = self.img.copy()
        result = imaugs.vstack([img, img])
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.width, img.width)
        self.assertEqual(result.height, img.height * 2)


if __name__ == "__main__":
    unittest.main()
