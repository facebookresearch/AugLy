#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from typing import Any, Dict, List, Optional, Tuple

from augly.image.transforms import BaseTransform
from PIL import Image

"""
Composition Operators:

Compose: the Compose operator was added here such that users
do not have to import `torchvision` in order to compose multiple
augmentations together. These operators work identically and either
can be used.

OneOf: the OneOf operator takes as input a list of transforms and
may apply (with probability p) one of the transforms in the list.
If a transform is applied, it is selected using the specified
probabilities of the individual transforms.

Example:

 >>> Compose([
 >>>     Blur(),
 >>>     ColorJitter(saturation_factor=1.5),
 >>>     OneOf([
 >>>         OverlayOntoScreenshot(),
 >>>         OverlayEmoji(),
 >>>         OverlayText(),
 >>>     ]),
 >>> ])
"""


class BaseComposition:
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        @param transforms: a list of transforms

        @param p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, (BaseTransform, BaseComposition)
            ), "Expected instances of type `BaseTransform` or `BaseComposition` for variable `transforms`"  # noqa: B950
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"

        self.transforms = transforms
        self.p = p


class Compose(BaseComposition):
    def __call__(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Applies the list of transforms in order to the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies the type of bounding box that was passed in in `bboxes`.
            Must specify `bbox_type` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        if random.random() > self.p:
            return image

        for transform in self.transforms:
            image = transform(
                image, metadata=metadata, bboxes=bboxes, bbox_format=bbox_format
            )

        return image


class OneOf(BaseComposition):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        @param transforms: a list of transforms to select from; one of which
            will be chosen to be applied to the media

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Applies one of the transforms to the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies the type of bounding box that was passed in in `bboxes`.
            Must specify `bbox_type` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        if random.random() > self.p:
            return image

        transform = random.choices(self.transforms, self.transform_probs)[0]
        return transform(
            image, force=True, metadata=metadata, bboxes=bboxes, bbox_format=bbox_format
        )
