#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import List, Optional, Union

import numpy as np
from augly.text.augmenters.utils import (
    rejoin_words_and_whitespace,
    split_words_on_whitespace,
)


class CaseChanger:
    def __init__(self, case: str, seed: Optional[int]):
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.case = case

    def change(self, text: str, case: Optional[str] = None) -> str:
        case = case or self.case
        if case == "lower":
            return text.lower()
        elif case == "upper":
            return text.upper()
        elif case == "title":
            return text.title()

        return self.change(text, self.rng.choice(["lower", "upper", "title"]))


class CaseAugmenter:
    """
    Augmenter that changes the case characters, words, or entire texts depending on the
    given 'granularity'.

    The 'cadence' option are used to adjust the frequency of the case changes, to avoid
    detection by automated deobfuscation techniques. If 'cadence' is not an integer
    (e.g. 2.5), the case will be changed on *average* about every 2.5 chars.
    """

    def __init__(
        self,
        case: str,
        granularity: str,
        cadence: float = 1.0,
        seed: Optional[int] = 10,
    ):
        assert granularity in {
            "char",
            "word",
            "all",
        }, "Must set 'granularity' to either 'word' or 'all'."
        assert cadence >= 1.0, "Must set 'cadence' to be no less than 1.0."
        assert case in {
            "lower",
            "upper",
            "title",
            "random",
        }, "'case' must be one of: lower, upper, title, random"

        self.case_changer = CaseChanger(case, seed)
        self.granularity = granularity
        self.cadence = cadence

    def change_case_all(self, text: str) -> str:
        return self.case_changer.change(text)

    def change_case_chars(self, text: str) -> str:
        num_change_case = int((len(text) + self.cadence - 1) // self.cadence)
        change_case_idxs = {math.ceil(i * self.cadence) for i in range(num_change_case)}
        augmented_chars = [
            self.case_changer.change(str(c)) if i in change_case_idxs else str(c)
            for i, c in enumerate(text)
        ]
        return "".join(augmented_chars)

    def change_case_words(self, text: str) -> str:
        words, spaces = split_words_on_whitespace(text)
        num_change_case = int((len(words) + self.cadence - 1) // self.cadence)
        change_case_idxs = {math.ceil(i * self.cadence) for i in range(num_change_case)}
        augmented_words = [
            self.case_changer.change(word) if i in change_case_idxs else word
            for i, word in enumerate(words)
        ]
        return rejoin_words_and_whitespace(augmented_words, spaces)

    def augment(self, texts: Union[str, List[str]]) -> List[str]:
        texts = texts if isinstance(texts, list) else [texts]

        if self.granularity == "all":
            return [self.change_case_all(text) for text in texts]
        elif self.granularity == "char":
            return [self.change_case_chars(text) for text in texts]

        # self.granularity == "word"
        return [self.change_case_words(text) for text in texts]
