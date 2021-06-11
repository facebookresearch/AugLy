#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

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
        results = []
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
                if i not in aug_word_idxes:
                    results.append(token)
                    continue

                results.append(self.flip_text(token))

        elif self.granularity == "char":
            all_chars = [char for token in tokens for char in list(token)]
            aug_char_idxes = self.generate_aug_idxes(all_chars)
            char_idx = 0

            for token in tokens:
                result = ""
                chars = list(token)

                for char in chars:
                    if char_idx not in aug_char_idxes:
                        result += char
                    else:
                        result += _flip(char)
                    char_idx += 1

                results.append(result)

        return detokenize(results)
