#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
from typing import Any, Dict, List, Optional, Union

from augly.text.augmenters.utils import (
    detokenize,
    get_aug_idxes,
    tokenize,
)
from augly.utils import pathmgr
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class WordReplacement(object):
    def __init__(self, mapping: Optional[Union[str, Dict[str, Any]]]):
        if isinstance(mapping, str):
            local_mapping_path = pathmgr.get_local_path(mapping)
            with open(local_mapping_path) as json_file:
                self.mapping = {k.lower(): v for k, v in json.load(json_file).items()}
        elif isinstance(mapping, Dict):
            self.mapping = mapping
        else:
            self.mapping = {}


    def replace(self, word: str) -> str:
        new_word = self.mapping.get(word, None) or self.mapping.get(word.lower(), None)
        if new_word is not None and word[0].isupper():
            new_word = new_word.capitalize()
        return new_word or word


class WordReplacementAugmenter(WordAugmenter):
    """Augmenter that replaces words based on a given mapping"""

    def __init__(
        self,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        mapping: Optional[Union[str, Dict[str, Any]]],
        priority_words: Optional[List[str]],
        ignore_words: Optional[List[str]],
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
            set(ignore_words) if ignore_words is not None else set()
        )

    def get_mapping(
        self, mapping: Optional[Union[str, Dict[str, Any]]]
    ) -> WordReplacement:
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

        for t_i, token in enumerate(tokens):
            if t_i not in aug_word_idxes:
                results.append(token)
                continue

            results.append(self.word_mapping.replace(token))

        return detokenize(results)
