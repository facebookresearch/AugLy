#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging

import cv2
import numpy as np
from augly.video.augmenters.cv2.base_augmenter import BaseCV2Augmenter
from augly.video.augmenters.cv2.shapes import VideoDistractorByShapes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VideoDistractorByDots(BaseCV2Augmenter):
    def __init__(
        self, num_dots: int, dot_type: str, random_movement: bool = True, **kwargs
    ) -> None:
        assert num_dots > 0, "Number of dots must be greater than zero"
        assert dot_type in [
            "colored",
            "blur",
        ], "Dot type must be set to None or to 'colored' or 'blur'"

        super().__init__(num_dots, random_movement, **kwargs)

        self.num_dots = num_dots
        self.dot_type = dot_type
        self.shapes_distractor = None

        if self.dot_type == "colored":
            self.shapes_distractor = VideoDistractorByShapes(
                num_dots,
                shape_type="circle",
                colors=[(0, 0, 0), (91, 123, 166)],
                thickness=2,
                radius=0.001,
                random_movement=random_movement,
            )

    def add_blurred_dots(self, raw_frame: np.ndarray) -> np.ndarray:
        height, width = raw_frame.shape[:2]
        distract_frame = raw_frame.copy()

        for i in range(self.num_dots):
            fraction_x, fraction_y = self.get_origins(i)
            x = int(fraction_x * width)
            y = int(fraction_y * height)

            # I think that sometimes the random positioning of the dot goes
            # past the frame resulting in an error, but I can't repro this, so
            # try/catching for now
            try:
                dot_bbox = distract_frame[y : y + 10, x : x + 10]
                dot_bbox = cv2.GaussianBlur(dot_bbox, (111, 111), cv2.BORDER_DEFAULT)
                distract_frame[y : y + 10, x : x + 10] = dot_bbox
            except Exception as e:
                logger.warning(f"Exception while adding Gaussian dot distractor: {e}")

        return distract_frame

    # overrides abstract method of base class
    def apply_augmentation(self, raw_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds random dot distracts (in various colors and positions) to each frame

        @param raw_frame: raw, single RGB/Gray frame

        @returns: the augmented frame
        """
        assert (raw_frame.ndim == 3) and (
            raw_frame.shape[2] == 3
        ), "VideoDistractorByDots only accepts RGB images"

        if self.dot_type == "colored":
            return self.shapes_distractor.apply_augmentation(raw_frame, **kwargs)

        return self.add_blurred_dots(raw_frame)
