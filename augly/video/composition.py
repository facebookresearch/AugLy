#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
import shutil
from typing import List, Optional

from augly.video.transforms import VidAugBaseClass
from augly.video.helpers import validate_input_and_output_paths


"""
Composition Operators:

Compose: identical to the Compose object provided by the torchvision
library, this class provides a similar experience for applying multiple
transformations onto a video

OneOf: the OneOf operator takes as input a list of transforms and
may apply (with probability p) one of the transforms in the list.
If a transform is applied, it is selected using the specified
probabilities of the individual transforms.

Example:

 >>> Compose([
 >>>     IGFilter(),
 >>>     ColorJitter(saturation_factor=1.5)
 >>>     OneOf([
 >>>         ScreenshotOverlay(),
 >>>         EmojiOverlay(),
 >>>         TextOverlay(),
 >>>     ]),
 >>> ])
"""


class BaseComposition(VidAugBaseClass):
    def __init__(self, transforms: List[VidAugBaseClass], p: float = 1.0):
        """
        @param transforms: a list of transforms

        @param p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, VidAugBaseClass
            ), "Expected instances of type 'VidAugBaseClass' for parameter 'transforms'"

        super().__init__(p)
        self.transforms = transforms


class Compose(BaseComposition):
    def __call__(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Applies the list of transforms in order to the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten
        """
        video_path, output_path = validate_input_and_output_paths(
            video_path, output_path
        )

        if video_path != output_path:
            shutil.copy(video_path, output_path)

        for transform in self.transforms:
            transform(output_path)


class OneOf(BaseComposition):
    def __init__(self, transforms: List[VidAugBaseClass], p: float = 1.0):
        """
        @param transforms: a list of transforms to select from; one of which will
            be chosen to be applied to the video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Applies one of the transforms to the video (with probability p)

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten
        """
        if random.random() > self.p:
            return None

        transform = random.choices(self.transforms, self.transform_probs)[0]
        return transform(video_path, output_path, force=True)
