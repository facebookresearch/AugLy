#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Any, Dict, List, Optional, Union

from augly.text.transforms import BaseTransform


"""
Composition Operators:

Compose: identical to the Compose object provided by the torchvision
library, this class provides a similar experience for applying multiple
transformations onto text

OneOf: the OneOf operator takes as input a list of transforms and
may apply (with probability p) one of the transforms in the list.
If a transform is applied, it is selected using the specified
probabilities of the individual transforms.

Example:

 >>> Compose([
 >>>     InsertPunctuationChars(),
 >>>     ReplaceFunFonts(),
 >>>     OneOf([
 >>>         ReplaceSimilarChars(),
 >>>         SimulateTypos(),
 >>>     ]),
 >>> ])
"""


class BaseComposition(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        @param transforms: a list of transforms

        @param p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, BaseTransform
            ), "Expected instances of type 'BaseTransform' for parameter 'transforms'"

        super().__init__(p)
        self.transforms = transforms


class Compose(BaseComposition):
    def __call__(
        self,
        texts: Union[str, List[str]],
        seed: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Applies the list of transforms in order to the text

        @param texts: a string or a list of text documents to be augmented

        @param seed: if provided, the random seed will be set to this before calling
            the transform

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        if seed is not None:
            random.seed(seed)

        texts = [texts] if isinstance(texts, str) else texts
        for transform in self.transforms:
            texts = transform(texts, metadata=metadata)

        return texts


class OneOf(BaseComposition):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        @param transforms: a list of transforms to select from; one of which will
            be chosen to be applied to the text

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(
        self,
        texts: Union[str, List[str]],
        force: bool = False,
        seed: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        @param texts: a string or a list of text documents to be augmented

        @param force: if set to True, the transform will be applied. Otherwise,
            application is determined by the probability set

        @param seed: if provided, the random seed will be set to this before calling
            the transform

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        if seed is not None:
            random.seed(seed)

        texts = [texts] if isinstance(texts, str) else texts

        if random.random() > self.p:
            return texts

        transform = random.choices(self.transforms, self.transform_probs)[0]
        return transform(texts, force=True, metadata=metadata)
