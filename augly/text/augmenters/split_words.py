#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import List, Optional

from augly.text.augmenters.utils import detokenize, get_aug_idxes, tokenize
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class SplitWordsAugmenter(WordAugmenter):
    """Augmenter that splits words into two words"""

    def __init__(
        self,
        min_char: int,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        priority_words: Optional[List[str]],
    ):
        assert min_char >= 2, "Expected min_char to be greater than or equal to 2"

        super().__init__(
            action=Action.SPLIT,
            aug_min=aug_word_min,
            aug_max=aug_word_max,
            aug_p=aug_word_p,
        )
        self.min_char = min_char
        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )

    def split(self, data: str) -> str:
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
