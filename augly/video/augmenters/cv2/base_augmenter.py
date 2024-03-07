#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Implementation of base class for video distractors

- Interface:
    - `augment(self, video_temp_path: str, fps: float, **kwargs)`:
    extracts frames from the video, and turns original set of frames into
    augmented frames by mapping each of them using `apply_augmentation` operators.
    the path to the temporary directory containing the augmented frames is returned

- Methods to override
    - `apply_augmentation(self, raw_frame: np.ndarray, **kwargs)`:
    applies the augmentation to each frame
"""

import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from augly import utils
from augly.video.helpers import extract_frames_to_dir


class BaseCV2Augmenter(ABC):
    def __init__(
        self,
        num_dist: int = 0,
        random_movement: bool = True,
        topleft: Optional[Tuple[float, float]] = None,
        bottomright: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        assert type(random_movement) == bool, "Random movement must be set to a boolean"

        assert topleft is None or all(
            0.0 <= t <= 1.0 for t in topleft
        ), "Topleft must be in the range [0, 1]"
        assert bottomright is None or all(
            0.0 <= b <= 1.0 for b in bottomright
        ), "Bottomright must be in the range [0, 1]"

        self.random_movement = random_movement

        if self.random_movement:
            self.origins = BaseCV2Augmenter.random_origins(topleft, bottomright)
        else:
            top, left = topleft or (0.01, 0.01)
            bottom, right = bottomright or (0.99, 0.99)

            y_vals = random.choices(np.linspace(top, bottom, num=15), k=num_dist)
            random.shuffle(y_vals)

            self.origins = [
                BaseCV2Augmenter.moving_origins(left, y_val, left, right)
                for y_val in y_vals
            ]

    def augment(self, video_temp_path: str, fps: float, **kwargs) -> str:
        """
        Augment a set of frames by adding distractors to each by mapping each
        frame with `apply_augmentation` method

        @param video_temp_path: local temp path of the video to be augmented

        @param kwargs: parameters to pass into apply_augmentation

        @returns: path to the temp directory containing augmented frames
        """
        video_temp_dir = Path(video_temp_path).parent
        frame_temp_dir = os.path.join(video_temp_dir, "raw_frames_distractor")
        os.mkdir(frame_temp_dir)
        aug_frame_temp_dir = os.path.join(video_temp_dir, "aug_frames_distractor")
        os.mkdir(aug_frame_temp_dir)
        extract_frames_to_dir(video_temp_path, frame_temp_dir)

        for raw_frame_file in os.listdir(frame_temp_dir):
            frame = cv2.imread(os.path.join(frame_temp_dir, raw_frame_file))
            aug_frame = self.apply_augmentation(frame, **kwargs)
            cv2.imwrite(os.path.join(aug_frame_temp_dir, raw_frame_file), aug_frame)

        shutil.rmtree(frame_temp_dir)
        return aug_frame_temp_dir

    @abstractmethod
    def apply_augmentation(self, raw_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the specific augmentation to a single frame

        @param raw_frame: raw, single RGB/Gray frame

        @returns: the augmented frame
        """
        raise NotImplementedError("Implement apply_augmentation method")

    def get_origins(self, index: int) -> Tuple[float, float]:
        if self.random_movement:
            return next(self.origins)

        return next(self.origins[index])

    @staticmethod
    def random_origins(
        topleft: Optional[Tuple[float, float]],
        bottomright: Optional[Tuple[float, float]],
    ) -> Iterator[Tuple[float, float]]:
        top, left = topleft or (0.01, 0.01)
        bottom, right = bottomright or (0.99, 0.99)
        while True:
            yield (random.uniform(left, right), random.uniform(top, bottom))

    @staticmethod
    def moving_origins(
        x_start: float, y_val: float, x_min: float, x_max: float
    ) -> Iterator[Tuple[float, float]]:
        x_curr = x_start
        while True:
            yield x_curr, y_val
            x_curr += 0.02
            x_curr = x_curr if x_curr <= x_max else x_min

    @staticmethod
    def random_colors(
        colors: Optional[List[Tuple[int, int, int]]]
    ) -> Iterator[Tuple[int, int, int]]:
        if colors is not None:
            assert type(colors) == list, "Expected type 'List' for colors variable"
            for color in colors:
                utils.validate_rgb_color(color)

        while True:
            if colors:
                color = colors[random.randint(0, len(colors) - 1)]
            else:
                color = (
                    random.randint(0, 255),  # R
                    random.randint(0, 255),  # G
                    random.randint(0, 255),  # B
                )
            yield color
