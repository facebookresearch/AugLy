#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision.transforms as transforms  # @manual
from augly import image as imaugs
from augly.tests import ImageAugConfig
from augly.utils import pathmgr
from PIL import Image

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
