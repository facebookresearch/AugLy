#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from augly.video.augmenters.cv2.base_augmenter import BaseCV2Augmenter
from augly.video.augmenters.cv2.shapes import VideoDistractorByShapes
from augly.video.augmenters.cv2.dots import VideoDistractorByDots
from augly.video.augmenters.cv2.text import VideoDistractorByText


__all__ = [
    "BaseCV2Augmenter",
    "VideoDistractorByText",
    "VideoDistractorByShapes",
    "VideoDistractorByDots",
]
