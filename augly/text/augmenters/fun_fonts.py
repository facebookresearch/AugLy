#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
import random
from typing import List, Optional, Union

from augly.text.augmenters.utils import detokenize, get_aug_idxes, tokenize
from augly.utils import pathmgr
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug import Augmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class FunFontsAugmenter(Augmenter):
    def __init__(
        self,
        granularity: str,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        vary_fonts: bool,
        fonts_path: str,
        priority_words: Optional[List[str]],
    ):
        assert granularity in [
            "char",
            "word",
            "all",
        ], "Granularity must be either char, word, or all"
        assert (
            0 <= aug_min <= aug_max
        ), "aug_min must be non-negative and aug_max must be greater than or equal to aug_min"
        assert 0 <= aug_p <= 1, "aug_p must be a value in the range [0, 1]"

        super().__init__(
            name="FunFontsAugmenter",
            action=Action.SUBSTITUTE,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )
        self.fonts = self.load_fonts(fonts_path)
        self.aug_p = aug_p
        self.granularity = granularity
        self.vary_fonts = vary_fonts
        self.priority_words = (
            set(priority_words) if priority_words is not None else priority_words
        )

    def load_fonts(self, fonts_path: str) -> List[Union[str, dict]]:
        """
        Loads the fonts from a json file iopath uri

        @returns mapping: the key corresponds to the font name; the value is
            a str if the font is the same for any replacement and the value is
            a dict[str, str] if the font maps every letter to a new font letter
        """
        local_fonts_path = pathmgr.get_local_path(fonts_path)

        with open(local_fonts_path, encoding='utf-8') as text_file:
            font_dictionary = json.load(text_file)
            return list(font_dictionary.values())

        return []

    @classmethod
    def clean(cls, data: Union[List[str], str]) -> Union[str, List[str]]:
        if isinstance(data, list):
            return [d.strip() for d in data]

        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset: List[str], data: str) -> bool:
        return data in dataset

    def apply_font(self, text: str, font: Union[str, dict], method: str) -> str:
        assert (
            method in Method.getall()
        ), "Expected 'method' to be a value defined in nlpaug.util.method.Method"

        if isinstance(font, str):
            return font.join(text) + font if method == Method.WORD else text + font

        if isinstance(font, dict):
            return (
                "".join([font.get(char, char) for char in text])
                if method == Method.WORD
                else font.get(text, text)
            )

    def substitute(self, data: str) -> str:
        tokens = tokenize(data)
        font = random.sample(self.fonts, 1)[0]
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

                if self.vary_fonts:
                    font = random.sample(self.fonts, 1)[0]

                results.append(self.apply_font(token, font, Method.WORD))

        elif self.granularity == "char":
            all_chars = [char for token in tokens for char in list(token)]
            aug_char_idxes = set(self.generate_aug_idxes(all_chars))
            char_idx = 0

            for token in tokens:
                result = ""
                chars = list(token)

                for char in chars:
                    if self.vary_fonts:
                        font = random.sample(self.fonts, 1)[0]
                    if char_idx not in aug_char_idxes:
                        result += char
                    else:
                        result += self.apply_font(char, font, Method.CHAR)
                    char_idx += 1

                results.append(result)

        else:
            if random.random() < self.aug_p:
                for token in tokens:
                    results.append(self.apply_font(token, font, Method.WORD))
            else:
                results = tokens

        return detokenize(results)
