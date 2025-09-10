#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from typing import Any

from augly.text.augmenters.utils import detokenize, get_aug_idxes, tokenize
from augly.utils import pathmgr
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
# pyre-fixme[21]: Could not find name `WordAugmenter` in `nlpaug.augmenter.word`.
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class WordReplacement:
    def __init__(self, mapping: str | dict[str, Any] | None):
        if isinstance(mapping, str):
            local_mapping_path = pathmgr.get_local_path(mapping)
            with open(local_mapping_path) as json_file:
                self.mapping = {k.lower(): v for k, v in json.load(json_file).items()}
        elif isinstance(mapping, dict):
            self.mapping = mapping
        else:
            self.mapping = {}

    def replace(self, word: str) -> tuple[str, bool]:
        new_word = self.mapping.get(word, None) or self.mapping.get(word.lower(), None)
        if new_word is not None and word[0].isupper():
            new_word = new_word.capitalize()
        return (new_word, True) if new_word else (word, False)


class WordReplacementAugmenter(WordAugmenter):
    """Augmenter that replaces words based on a given mapping"""

    def __init__(
        self,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        mapping: str | dict[str, Any] | None,
        priority_words: list[str] | None,
        ignore_words: list[str] | None,
    ):
        super().__init__(
            action=Action.SUBSTITUTE,
            aug_min=aug_word_min,
            aug_max=aug_word_max,
            aug_p=aug_word_p,
        )
        self.word_mapping = self.get_mapping(mapping)
        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )
        self.ignore_words = (
            {word.lower() for word in ignore_words}
            if ignore_words is not None
            else set()
        )

    def get_mapping(self, mapping: str | dict[str, Any] | None) -> WordReplacement:
        return WordReplacement(mapping)

    def substitute(self, data: str) -> str:
        """
        Returns a text where random words are replaced using the specified mapping

        @param data: the text to which the word substitution will be applied
        """
        results = []
        tokens = tokenize(data)
        aug_word_cnt = self._generate_aug_cnt(
            len(tokens), self.aug_min, self.aug_max, self.aug_p
        )
        filtered_word_idxes = self.pre_skip_aug(tokens)

        if self.priority_words is None:
            self.priority_words = self.word_mapping.mapping.keys()

        aug_word_idxes = set(
            get_aug_idxes(
                self,
                tokens,
                filtered_word_idxes,
                aug_word_cnt,
                Method.WORD,
            )
        )

        if not aug_word_idxes:
            return data

        is_diff = False
        for t_i, token in enumerate(tokens):
            if t_i not in aug_word_idxes:
                results.append(token)
                continue

            new_token, has_changed = self.word_mapping.replace(token)
            is_diff = is_diff or has_changed
            results.append(new_token)

        return detokenize(results) if is_diff else data
