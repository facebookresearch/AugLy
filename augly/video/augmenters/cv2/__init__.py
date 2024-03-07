#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from augly.video.augmenters.cv2.base_augmenter import BaseCV2Augmenter
from augly.video.augmenters.cv2.dots import VideoDistractorByDots
from augly.video.augmenters.cv2.shapes import VideoDistractorByShapes
from augly.video.augmenters.cv2.text import VideoDistractorByText


__all__ = [
    "BaseCV2Augmenter",
    "VideoDistractorByText",
    "VideoDistractorByShapes",
    "VideoDistractorByDots",
]
