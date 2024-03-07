#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Union

from augly.text.augmenters.utils import (
    detokenize,
    get_aug_idxes,
    tokenize,
    UPSIDE_DOWN_CHAR_MAPPING,
)
from augly.utils.libsndfile import install_libsndfile

install_libsndfile()
from nlpaug.augmenter.word import Augmenter  # @manual
from nlpaug.util import Action, Method  # @manual


def _flip(c: str) -> str:
    if c in UPSIDE_DOWN_CHAR_MAPPING:
        return UPSIDE_DOWN_CHAR_MAPPING[c]
    else:
        return c


class UpsideDownAugmenter(Augmenter):
    """Augmenter that flips the text"""

    def __init__(self, granularity: str, aug_min: int, aug_max: int, aug_p: float):
        assert granularity in [
            "char",
            "word",
            "all",
        ], "Granularity must be either char, word, or all"
        assert (
            0 <= aug_min <= aug_max
        ), "aug_min must be non-negative and aug_max must be greater than or equal to aug_min"
        assert 0 <= aug_p <= 1, "aug_p must be a value in the range [0, 1]"

        self.granularity = granularity
        super().__init__(
            name="UpsideDownAugmenter",
            action=Action.INSERT,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )

    @classmethod
    def clean(cls, data: Union[List[str], str]) -> Union[str, List[str]]:
        if isinstance(data, list):
            return [d.strip() for d in data]

        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: str) -> bool:
        return data in dataset

    def flip_text(self, text: str) -> str:
        return "".join([_flip(c) for c in reversed(text)])

    def insert(self, data: str) -> str:
        if self.granularity == "all":
            return self.flip_text(data)

        tokens = tokenize(data)

        if self.granularity == "word":
            aug_word_cnt = self._generate_aug_cnt(
                len(tokens), self.aug_min, self.aug_max, self.aug_p
            )
            aug_word_idxes = set(
                get_aug_idxes(
                    self, tokens, list(range(len(tokens))), aug_word_cnt, Method.WORD
                )
            )

            for i, token in enumerate(tokens):
                if i in aug_word_idxes:
                    tokens[i] = self.flip_text(token)

        elif self.granularity == "char":
            all_chars = [char for token in tokens for char in list(token)]
            aug_char_idxes = self.generate_aug_idxes(all_chars)
            char_idx = 0

            for t_i, token in enumerate(tokens):
                chars = list(token)

                for c_i, char in enumerate(chars):
                    if char_idx in aug_char_idxes:
                        chars[c_i] = _flip(char)
                    char_idx += 1

                tokens[t_i] = "".join(chars)

        return detokenize(tokens)
