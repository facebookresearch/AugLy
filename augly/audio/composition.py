#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from augly.audio.transforms import BaseTransform


"""
Composition Operators:

Compose: identical to the Compose object provided by the torchvision
library, this class provides a similar experience for applying multiple
transformations onto audio

OneOf: the OneOf operator takes as input a list of transforms and
may apply (with probability p) one of the transforms in the list.
If a transform is applied, it is selected using the specified
probabilities of the individual transforms.

Example:

 >>> Compose([
 >>>     Clip(duration_factor=0.5),
 >>>     VolumeChange(volume_db=10.0),
 >>>     OneOf([
 >>>        PitchShift(n_steps=4.0),
 >>>        TimeStretch(rate=1.5),
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
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies the list of transforms in order to the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        for transform in self.transforms:
            audio, sample_rate = transform(audio, sample_rate, metadata)
        return audio, sample_rate


class OneOf(BaseComposition):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        @param transforms: a list of transforms to select from; one of which will
            be chosen to be applied to the audio

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies one of the transforms to the audio (with probability p)

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        if random.random() > self.p:
            return audio, sample_rate

        transform = random.choices(self.transforms, self.transform_probs)[0]
        return transform(audio, sample_rate, metadata, force=True)
