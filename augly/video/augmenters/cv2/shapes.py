#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Optional, List, Tuple, Iterator

import cv2
import numpy as np
from augly.video.augmenters.cv2 import BaseCV2Augmenter


class VideoDistractorByShapes(BaseCV2Augmenter):
    def __init__(
        self,
        num_shapes: int,
        shape_type: Optional[str] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: Optional[int] = None,
        radius: Optional[float] = None,
        random_movement: bool = True,
        topleft: Optional[Tuple[float, float]] = None,
        bottomright: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> None:
        assert num_shapes > 0, "Number of shapes must be greater than zero"
        assert shape_type is None or shape_type in [
            "circle",
            "rectangle",
        ], "Shape type must be set to None or to 'circle' or 'rectangle'"
        assert (
            thickness is None or type(thickness) == int
        ), "Invalid thickness provided: must be set to None or an integer"

        super().__init__(num_shapes, random_movement, topleft, bottomright, **kwargs)

        self.num_shapes = num_shapes
        self.shape_type = VideoDistractorByShapes.random_shape_type(shape_type)
        self.colors = BaseCV2Augmenter.random_colors(colors)
        self.thickness = VideoDistractorByShapes.random_thickness(thickness)
        self.radius = VideoDistractorByShapes.random_radius(radius)

    # overrides abstract method of base class
    def apply_augmentation(self, raw_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds shape distracts (in various colors and positions) to each frame

        @param raw_frame: raw, single RGB/Gray frame

        @returns: the augumented frame
        """
        assert (raw_frame.ndim == 3) and (
            raw_frame.shape[2] == 3
        ), "VideoDistractorByShapes only accepts RGB images"
        height, width = raw_frame.shape[:2]
        distract_frame = raw_frame.copy()

        for i in range(self.num_shapes):
            shape_type = next(self.shape_type)
            color = next(self.colors)
            thickness = next(self.thickness)
            fraction_x, fraction_y = self.get_origins(i)
            x = int(fraction_x * width)
            y = int(fraction_y * height)

            if shape_type == "circle":
                smaller_side = min(height, width)
                radius = int(next(self.radius) * smaller_side)
                cv2.circle(distract_frame, (x, y), radius, color, thickness)

            if shape_type == "rectangle":
                fraction_x, fraction_y = self.get_origins(i)
                x_2 = int(fraction_x * width)
                y_2 = int(fraction_y * height)
                cv2.rectangle(distract_frame, (x, y), (x_2, y_2), color, thickness)

        return distract_frame

    @staticmethod
    def random_shape_type(shape_type: Optional[str]) -> Iterator[str]:
        shapes = ["circle", "rectangle"]

        while True:
            yield shape_type if shape_type else shapes[
                random.randint(0, len(shapes) - 1)
            ]

    @staticmethod
    def random_thickness(thickness: Optional[int]) -> Iterator[int]:
        while True:
            yield thickness or random.randint(-1, 5)

    @staticmethod
    def random_radius(radius: Optional[float]) -> Iterator[float]:
        if radius:
            radius = radius if radius < 0.5 else radius / 2
        while True:
            yield radius or (random.random() / 2)
