#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
import torchvision.transforms as transforms  # @manual
from PIL import Image

import augly.image as imaugs
from augly.tests import ImageAugConfig
from augly.utils import pathmgr

COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 1.2,
    "saturation_factor": 1.4,
}

AUGMENTATIONS = [
    imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    imaugs.OneOf(
        [imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText()]
    ),
]

TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])


class ComposeAugmentationsTestCase(unittest.TestCase):
    def test_torchvision_compose_compability(self) -> None:
        config = ImageAugConfig()

        image_path, image_file = config.get_input_path()
        local_img_path = pathmgr.get_local_path(image_path)

        image = Image.open(local_img_path)
        tsfm_image = TENSOR_TRANSFORMS(image)
        self.assertIsInstance(tsfm_image, torch.Tensor)

    def test_augly_image_compose(self) -> None:
        config = ImageAugConfig()

        image_path, image_file = config.get_input_path()
        local_img_path = pathmgr.get_local_path(image_path)

        image = Image.open(local_img_path)
        tsfm_image = TRANSFORMS(image)
        self.assertIsInstance(tsfm_image, Image.Image)


if __name__ == "__main__":
    unittest.main()
