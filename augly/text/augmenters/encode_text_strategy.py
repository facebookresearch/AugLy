#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from abc import abstractmethod
from typing import List, Union

from augly.text.augmenters.utils import detokenize, Encoding, get_aug_idxes, tokenize
from nlpaug.augmenter.word import Augmenter
from nlpaug.util import Action, Method


class EncodeTextAugmentation(Augmenter):
    def __init__(
        self,
        name: str,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        encoder: Encoding = Encoding.BASE64,
        method: str = Method.SENTENCE,
    ):
        super().__init__(
            name=name,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            action=Action.SUBSTITUTE,
            method=method,
        )

        self.encoder = encoder
        self.method = method

    @classmethod
    def clean(cls, data: Union[str, List[str], None]) -> Union[str, List[str]]:
        if isinstance(data, str):
            return data
        elif isinstance(data, list):
            cleaned_data = [cls.clean(d) for d in data]
            if all(isinstance(d, str) for d in cleaned_data):
                # pyre-ignore
                return cleaned_data
            return "".join(str(d) for d in cleaned_data)
        elif data is None:
            return ""
        else:
            return str(data)

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: str) -> bool:
        return data in dataset

    @abstractmethod
    def encode(self, input_string: str) -> str:
        raise NotImplementedError

    def substitute(self, data: str) -> str:
        if self.method == Method.SENTENCE:
            return self.encode(data)

        tokens = tokenize(data)
        if not tokens:
            return ""

        if self.method == Method.WORD:
            augment_count = self._generate_aug_cnt(
                len(tokens), self.aug_min, self.aug_max, self.aug_p
            )
            to_augment = set(
                get_aug_idxes(
                    self, tokens, list(range(len(tokens))), augment_count, Method.WORD
                )
            )
            for i, token in enumerate(tokens):
                if i in to_augment:
                    tokens[i] = self.encode(token)

        elif self.method == Method.CHAR:
            for token_idx, token in enumerate(tokens):
                chars = list(token)
                augment_count = self._generate_aug_cnt(
                    len(chars), self.aug_min, self.aug_max, self.aug_p
                )
                to_augment = set(
                    get_aug_idxes(
                        self, chars, list(range(len(chars))), augment_count, Method.CHAR
                    )
                )
                for char_idx, char in enumerate(chars):
                    if char_idx in to_augment:
                        chars[char_idx] = self.encode(char)
                tokens[token_idx] = "".join(chars)
        return detokenize(tokens)
