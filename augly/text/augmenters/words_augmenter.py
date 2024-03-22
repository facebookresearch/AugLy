#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from typing import List, Optional

from augly.text.augmenters.utils import detokenize, get_aug_idxes, tokenize
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
# pyre-fixme[21]: Could not find name `WordAugmenter` in `nlpaug.augmenter.word`.
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Method  # @manual


class WordsAugmenter(WordAugmenter):
    """
    Augmenter that performs a given action at the word level.
    Valid actions are defined below as methods
    """

    def __init__(
        self,
        action: str,
        min_char: int,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        priority_words: Optional[List[str]],
    ):
        assert min_char >= 2, "Expected min_char to be greater than or equal to 2"

        super().__init__(
            action=action,
            aug_min=aug_word_min,
            aug_max=aug_word_max,
            aug_p=aug_word_p,
        )
        self.min_char = min_char
        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )

    def delete(self, data: str) -> str:
        """Augmenter that merges selected words with the following word"""
        results = []
        tokens = tokenize(data)
        aug_word_cnt = self._generate_aug_cnt(
            len(tokens), self.aug_min, self.aug_max, self.aug_p
        )

        # Skip last word in the sentence as merges occur with the following word
        filtered_word_idxes = self.pre_skip_aug(tokens[:-1])
        aug_word_idxes = set(
            get_aug_idxes(
                self,
                tokens,
                filtered_word_idxes,
                aug_word_cnt,
                Method.WORD,
                self.min_char,
            )
        )

        if not aug_word_idxes:
            return data

        t_i = 0
        while t_i < len(tokens):
            if t_i in aug_word_idxes and len(tokens[t_i + 1]) >= self.min_char:
                results.append(tokens[t_i] + tokens[t_i + 1])
                t_i += 1
            else:
                results.append(tokens[t_i])

            t_i += 1

        return detokenize(results)

    def split(self, data: str) -> str:
        """Augmenter that splits words in two"""
        results = []
        tokens = tokenize(data)
        aug_word_cnt = self._generate_aug_cnt(
            len(tokens), self.aug_min, self.aug_max, self.aug_p
        )
        filtered_word_idxes = self.pre_skip_aug(tokens)
        aug_word_idxes = set(
            get_aug_idxes(
                self,
                tokens,
                filtered_word_idxes,
                aug_word_cnt,
                Method.WORD,
                self.min_char,
            )
        )

        if not aug_word_idxes:
            return data

        for t_i, token in enumerate(tokens):
            if t_i not in aug_word_idxes:
                results.append(token)
                continue

            target_token = tokens[t_i]
            split_position = random.randint(1, len(target_token) - 1)
            first_token = target_token[:split_position]
            second_token = target_token[split_position:]
            results.extend([first_token, second_token])

        return detokenize(results)
