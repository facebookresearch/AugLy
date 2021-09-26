#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
from typing import List, Optional

from augly.text.augmenters.utils import (
    detokenize,
    get_aug_idxes,
    tokenize,
    validate_augmenter_params,
)
from augly.utils import pathmgr
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug.augmenter.char import KeyboardAug, RandomCharAug  # @manual
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class MisspellingReplacement(object):
    def __init__(self, misspelling_dict_path: str):
        local_misspelling_dict_path = pathmgr.get_local_path(misspelling_dict_path)
        with open(local_misspelling_dict_path) as json_file:
            self.dictionary = json.load(json_file)

    def replace(self, word: str) -> Optional[List[str]]:
        return self.dictionary.get(word, None) or self.dictionary.get(word.lower(), None)


class TypoAugmenter(WordAugmenter):
    """
    Augmenter that replaces words with typos. You can specify a typo_type: charmix, which
    does a combination of character-level modifications (delete, insert, substitute, &
    swap); keyboard, which swaps characters which those close to each other on the QWERTY
    keyboard; misspelling, which replaces words with misspellings defined in a dictionary
    file; or all, which will apply a random combination of all 4
    """

    def __init__(
        self,
        min_char: int,
        aug_char_min: int,
        aug_char_max: int,
        aug_char_p: float,
        aug_word_min: int,
        aug_word_max: int,
        aug_word_p: float,
        typo_type: str,
        misspelling_dict_path: Optional[str],
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

        assert (
            typo_type in ["all", "charmix", "keyboard", "misspelling"]
        ), "Typo type must be one of: all, charmix, keyboard, misspelling"

        super().__init__(
            action=Action.SUBSTITUTE,
            aug_min=aug_word_min,
            aug_max=aug_word_max,
            aug_p=aug_word_p,
        )

        self.augmenters, self.model = [], None
        if typo_type in ["all", "charmix"]:
            for action in [Action.DELETE, Action.INSERT, Action.SUBSTITUTE, Action.SWAP]:
                self.augmenters.append(
                    RandomCharAug(
                        action=action,
                        min_char=min_char,
                        aug_char_min=aug_char_min,
                        aug_char_max=aug_char_max,
                        aug_char_p=aug_char_p,
                        reverse_tokenizer=detokenize,
                        tokenizer=tokenize,
                    ),
                )

        if typo_type in ["all", "keyboard"]:
            self.augmenters.append(
                KeyboardAug(
                    min_char=min_char,
                    aug_char_min=aug_char_min,
                    aug_char_max=aug_char_max,
                    aug_char_p=aug_char_p,
                    include_upper_case=True,
                    reverse_tokenizer=detokenize,
                    tokenizer=tokenize,
                ),
            )

        if typo_type in ["all", "misspelling"]:
            assert (
                misspelling_dict_path is not None
            ), "'misspelling_dict_path' must be provided if 'typo_type' is 'all' or 'misspelling'"
            self.model = self.get_model(misspelling_dict_path)

        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )

    def align_capitalization(self, src_token: str, dest_token: str) -> str:
        if self.get_word_case(src_token) == "upper":
            return dest_token.upper()
        elif self.get_word_case(src_token) == "lower":
            return dest_token.lower()
        elif self.get_word_case(src_token) == "capitalize":
            return dest_token.capitalize()

        return dest_token

    def get_model(self, misspelling_dict_path: str) -> MisspellingReplacement:
        return MisspellingReplacement(misspelling_dict_path)

    def substitute(self, data: str) -> str:
        """
        Returns text where random words are typos

        @param data: the text where the word substitutions will occur
        """
        results = []
        tokens = self.tokenizer(data)
        aug_word_cnt = self._generate_aug_cnt(
            len(tokens), self.aug_min, self.aug_max, self.aug_p
        )
        filtered_word_idxes = self.skip_aug(self.pre_skip_aug(tokens), tokens)
        aug_word_idxes = set(
            get_aug_idxes(
                self, tokens, filtered_word_idxes, aug_word_cnt, Method.WORD
            )
        )

        for t_i, token in enumerate(tokens):
            if t_i not in aug_word_idxes:
                results.append(token)
                continue

            misspellings = self.model.replace(token) if self.model else None
            if misspellings:
                misspelling = self.sample(misspellings, 1)[0]
                results.append(self.align_capitalization(token, misspelling))
            elif len(self.augmenters) > 0:
                aug = self.sample(self.augmenters, 1)[0]
                new_token = aug.augment(token)
                results.append(self.align_capitalization(token, new_token))
            else:
                # If no misspelling is found in the dict & no other typo types are being
                # used, don't change the token
                results.append(token)

        return detokenize(results)

    def _tokenizer(self, text: str) -> List[str]:
        return tokenize(text)
