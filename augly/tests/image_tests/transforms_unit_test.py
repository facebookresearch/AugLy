#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import unittest

from augly import image as imaugs
from augly.tests.image_tests.base_unit_test import BaseImageUnitTest
from augly.utils import EMOJI_PATH, IMAGE_METADATA_PATH, IMG_MASK_PATH
from PIL import Image


class TransformsImageUnitTest(BaseImageUnitTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(IMAGE_METADATA_PATH, "r") as f:
            cls.metadata = json.load(f)

    def test_ApplyLambda(self):
        self.evaluate_class(imaugs.ApplyLambda(), fname="apply_lambda")

    def test_ApplyPILFilter(self):
        self.evaluate_class(imaugs.ApplyPILFilter(), fname="apply_pil_filter")

    def test_Blur(self):
        self.evaluate_class(imaugs.Blur(), fname="blur")

    def test_Brightness(self):
        self.evaluate_class(imaugs.Brightness(), fname="brightness")

    def test_ChangeAspectRatio(self):
        self.evaluate_class(imaugs.ChangeAspectRatio(), fname="change_aspect_ratio")

    def test_ClipImageSize(self):
        self.evaluate_class(
            imaugs.ClipImageSize(max_resolution=1500000), fname="clip_image_size"
        )

    def test_ColorJitter(self):
        self.evaluate_class(imaugs.ColorJitter(), fname="color_jitter")

    def test_Compose(self):
        random.seed(1)
        self.evaluate_class(
            imaugs.Compose(
                [
                    imaugs.Blur(),
                    imaugs.ColorJitter(saturation_factor=1.5),
                    imaugs.OneOf(
                        [
                            imaugs.OverlayOntoScreenshot(),
                            imaugs.OverlayEmoji(),
                            imaugs.OverlayText(),
                        ]
                    ),
                ]
            ),
            fname="compose",
        )

    def test_Contrast(self):
        self.evaluate_class(imaugs.Contrast(), fname="contrast")

    def test_ConvertColor(self):
        self.evaluate_class(
            imaugs.ConvertColor(mode="L"),
            fname="convert_color",
            check_mode=False,
        )

    def test_Crop(self):
        self.evaluate_class(imaugs.Crop(), fname="crop")

    def test_EncodingQuality(self):
        self.evaluate_class(
            imaugs.EncodingQuality(quality=30), fname="encoding_quality"
        )

    def test_Grayscale(self):
        self.evaluate_class(imaugs.Grayscale(), fname="grayscale")

    def test_HFlip(self):
        self.evaluate_class(imaugs.HFlip(), fname="hflip")

    def test_MaskedComposite(self):
        self.evaluate_class(
            imaugs.MaskedComposite(
                mask=IMG_MASK_PATH,
                transform_function=imaugs.Brightness(factor=0.1),
            ),
            fname="masked_composite",
        )

    @unittest.skip("Failing on some envs, will fix")
    def test_MemeFormat(self):
        self.evaluate_class(imaugs.MemeFormat(), fname="meme_format")

    def test_Opacity(self):
        self.evaluate_class(imaugs.Opacity(), fname="opacity")

    def test_OverlayEmoji(self):
        self.evaluate_class(imaugs.OverlayEmoji(), fname="overlay_emoji")

    def test_OverlayImage(self):
        self.evaluate_class(
            imaugs.OverlayImage(overlay=EMOJI_PATH, overlay_size=0.15, y_pos=0.8),
            fname="overlay_image",
        )

    def test_OverlayOntoBackgroundImage(self):
        self.evaluate_class(
            imaugs.OverlayOntoBackgroundImage(
                background_image=EMOJI_PATH, overlay_size=0.5, scale_bg=True
            ),
            fname="overlay_onto_background_image",
        )

    def test_OverlayOntoScreenshot(self):
        self.evaluate_class(
            imaugs.OverlayOntoScreenshot(resize_src_to_match_template=False),
            fname="overlay_onto_screenshot",
            metadata_exclude_keys=[
                "dst_bboxes",
                "dst_height",
                "dst_width",
                "intensity",
                "template_filepath",
            ],
        )

    def test_OverlayStripes(self):
        self.evaluate_class(imaugs.OverlayStripes(), fname="overlay_stripes")

    @unittest.skip("Failing on some envs, will fix")
    def test_OverlayText(self):
        text_indices = [5, 3, 1, 2, 1000, 221]
        self.evaluate_class(imaugs.OverlayText(text=text_indices), fname="overlay_text")

    def test_Pad(self):
        self.evaluate_class(imaugs.Pad(), fname="pad")

    def test_PadSquare(self):
        self.evaluate_class(imaugs.PadSquare(), fname="pad_square")

    def test_PerspectiveTransform(self):
        self.evaluate_class(
            imaugs.PerspectiveTransform(sigma=100.0), fname="perspective_transform"
        )

    def test_Pixelization(self):
        self.evaluate_class(imaugs.Pixelization(), fname="pixelization")

    def test_RandomAspectRatio(self):
        random.seed(1)
        self.evaluate_class(imaugs.RandomAspectRatio(), fname="RandomAspectRatio")

    def test_RandomBlur(self):
        random.seed(1)
        self.evaluate_class(imaugs.RandomBlur(), fname="RandomBlur")

    def test_RandomBrightness(self):
        random.seed(1)
        self.evaluate_class(imaugs.RandomBrightness(), fname="RandomBrightness")

    @unittest.skip("Failing on some envs, will fix")
    def test_RandomEmojiOverlay(self):
        random.seed(1)
        self.evaluate_class(
            imaugs.RandomEmojiOverlay(emoji_size=(0.15, 0.3)),
            fname="RandomEmojiOverlay",
        )

    def test_RandomNoise(self):
        self.evaluate_class(imaugs.RandomNoise(), fname="random_noise")

    def test_RandomPixelization(self):
        random.seed(1)
        self.evaluate_class(imaugs.RandomPixelization(), fname="RandomPixelization")

    def test_RandomRotation(self):
        random.seed(1)
        self.evaluate_class(imaugs.RandomRotation(), fname="RandomRotation")

    def test_Resize(self):
        self.evaluate_class(imaugs.Resize(resample=Image.BICUBIC), fname="resize")

    def test_Rotate(self):
        self.evaluate_class(imaugs.Rotate(), fname="rotate")

    def test_Saturation(self):
        self.evaluate_class(imaugs.Saturation(factor=0.5), fname="saturation")

    def test_Scale(self):
        self.evaluate_class(imaugs.Scale(), fname="scale")

    def test_Sharpen(self):
        self.evaluate_class(imaugs.Sharpen(factor=2.0), fname="sharpen")

    def test_ShufflePixels(self):
        self.evaluate_class(imaugs.ShufflePixels(factor=0.5), fname="shuffle_pixels")

    def test_Skew(self):
        self.evaluate_class(imaugs.Skew(), fname="skew")

    def test_VFlip(self):
        self.evaluate_class(imaugs.VFlip(), fname="vflip")


if __name__ == "__main__":
    unittest.main()
