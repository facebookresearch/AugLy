#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Union

from augly.text.augmenters.utils import (
    rejoin_words_and_whitespace,
    split_words_on_whitespace,
)


POP_DIRECTIONAL = u"\u202C"
RTL_OVERRIDE = u"\u202E"
LTR_OVERRIDE = u"\u202D"


class BidirectionalAugmenter(object):
    """
    Reverses words in a string (or the whole string), using bidirectional
    override marks so the rendered string is visually identical to the original.
    Preserves whitespace, including newlines.
    """

    def __init__(self, granularity: str, split_word: bool = False):
        assert granularity in [
            "word",
            "all",
        ], "Must set 'granularity' to either 'word' or 'all'."

        self.granularity = granularity
        self.split_word = split_word

    def rtl_flip(self, text: str, split_text: bool) -> str:
        if split_text and len(text) > 1:
            split_point = len(text) // 2
            return (
                text[:split_point]
                + RTL_OVERRIDE
                + text[split_point:][::-1]
                + POP_DIRECTIONAL
            )

        return RTL_OVERRIDE + text[::-1] + POP_DIRECTIONAL

    def rtl_flip_per_word(self, text: str) -> str:
        words, spaces = split_words_on_whitespace(text)
        augmented_words = []

        for word in words:
            augmented_words.append(self.rtl_flip(word, self.split_word))

        return LTR_OVERRIDE + rejoin_words_and_whitespace(augmented_words, spaces)

    def augment(self, texts: Union[str, List[str]]) -> List[str]:
        texts = texts if isinstance(texts, list) else [texts]

        if self.granularity == "all":
            return [self.rtl_flip(text, False) for text in texts]

        # otherwise, self.granularity == "word"
        return [self.rtl_flip_per_word(text) for text in texts]
