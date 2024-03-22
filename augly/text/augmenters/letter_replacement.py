#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from typing import List, Optional

from augly.text.augmenters.utils import (
    get_aug_idxes,
    LETTER_CHAR_MAPPING,
    rejoin_words_and_whitespace,
    split_words_on_whitespace,
    validate_augmenter_params,
)
from augly.utils import pathmgr
from augly.utils.libsndfile import install_libsndfile

install_libsndfile()
# pyre-fixme[21]: Could not find name `CharAugmenter` in `nlpaug.augmenter.char`.
from nlpaug.augmenter.char import CharAugmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class CharReplacement:
    def __init__(self, mapping_path: Optional[str]):
        if mapping_path:
            local_mapping_path = pathmgr.get_local_path(mapping_path)
            with open(local_mapping_path) as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = LETTER_CHAR_MAPPING

    def replace(self, character: str) -> List[str]:
        return (
            self.mapping[character.lower()]
            if character.lower() in self.mapping
            else [character]
        )


class LetterReplacementAugmenter(CharAugmenter):
    """Augmenter that replaces letters with similar mappings"""

    def __init__(
        self,
        min_char: int,
        aug_char_min: int,
        aug_char_max: int,
        aug_char_p: float,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        mapping_path: Optional[str],
        priority_words: Optional[List[str]],
    ):
        validate_augmenter_params(
            aug_char_min,
            aug_char_max,
            aug_char_p,
            aug_word_min,
            aug_word_max,
            aug_word_p,
        )

        super().__init__(
            action=Action.SUBSTITUTE,
            min_char=min_char,
            aug_char_min=aug_char_min,
            aug_char_max=aug_char_max,
            aug_char_p=aug_char_p,
            aug_word_min=aug_word_min,
            aug_word_max=aug_word_max,
            aug_word_p=aug_word_p,
        )

        self.letter_mapping = self.get_mapping(mapping_path)
        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )

    def get_mapping(self, mapping_path: Optional[str]) -> CharReplacement:
        return CharReplacement(mapping_path)

    def substitute(self, data: str) -> str:
        """
        Returns a text where random letters are replaced by the specified mapping

        @param data: the text where the letter substitution will be applied on
        """
        tokens, whitespaces = split_words_on_whitespace(data)
        aug_word_cnt = self._generate_aug_cnt(
            len(tokens), self.aug_word_min, self.aug_word_max, self.aug_word_p
        )
        filtered_word_idxes = self.skip_aug(self.pre_skip_aug(tokens), tokens)
        aug_word_idxes = set(
            get_aug_idxes(self, tokens, filtered_word_idxes, aug_word_cnt, Method.WORD)
        )

        for t_i, token in enumerate(tokens):
            if t_i not in aug_word_idxes:
                continue

            chars = list(token)
            aug_char_cnt = self._generate_aug_cnt(
                len(chars), self.aug_char_min, self.aug_char_max, self.aug_char_p
            )
            aug_char_idxes = (
                None
                if len(chars) < self.min_char
                else set(
                    get_aug_idxes(
                        self, chars, list(range(len(chars))), aug_char_cnt, Method.CHAR
                    )
                )
            )

            if not aug_char_idxes:
                continue

            for c_i, char in enumerate(chars):
                if c_i in aug_char_idxes:
                    chars[c_i] = self.sample(self.letter_mapping.replace(char), 1)[0]

            tokens[t_i] = "".join(chars)

        return rejoin_words_and_whitespace(tokens, whitespaces)
