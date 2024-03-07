#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import random
from typing import List, Union

from augly.text.augmenters.utils import (
    rejoin_words_and_whitespace,
    split_words_on_whitespace,
)


# Not meant to be an exhaustive list
CHARACTER_TYPES = {
    "zero_width": [
        "\u200B",  # Zero-Width Space
        "\u200C",  # Zero-Width Non-Joiner
        "\u200D",  # Zero-Width Joiner
        "\u2060",  # Word Joiner
        "\u2061",  # Function Application
        "\u2062",  # Invisible Times
        "\u2063",  # Invisible Separator
        "\u2064",  # Invisible Plus
    ],
    "whitespace": [
        " ",  # Space
        "\t",  # Horizontal Tab
        "\n",  # Newline
        "\r",  # Carriage Return
        "\v",  # Vertical Tab
        "\f",  # Feed
    ],
    "punctuation": [".", "?", "!", ",", ";", ":", "-", "'", "..."],
}


class InsertionAugmenter:
    """
    Inserts various types of characters (including zero-width or punctuation),
    which in turn breaks up the text.

    The 'cadence' and 'vary_chars' options are used to adjust the frequency and
    pattern of the characters, to avoid detection by automated deobfuscation techniques.

    If 'cadence' is not an integer (say, 2.5), a char will be added
    on *average* about every 2.5 chars.
    """

    def __init__(
        self,
        char_type: str,
        granularity: str,
        cadence: float = 1.0,
        vary_chars: bool = False,
    ):
        char_types = set(CHARACTER_TYPES.keys())
        assert char_type in char_types, f"Must set 'char_type' to one of: {char_types}"
        assert granularity in {
            "word",
            "all",
        }, "Must set 'granularity' to either 'word' or 'all'."
        assert cadence >= 1.0, "Must set 'cadence' to be no less than 1.0."

        self.char_set = CHARACTER_TYPES[char_type]
        self.granularity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def insert_chars(self, text: str) -> str:
        splits = int((len(text) + self.cadence - 1) // self.cadence)
        split_text = [
            text[math.ceil(i * self.cadence) : math.ceil((i + 1) * self.cadence)]
            for i in range(splits)
        ]

        if not self.vary_chars:
            return random.choice(self.char_set).join(split_text)

        separators = random.choices(self.char_set, k=len(split_text))
        # pyre-fixme[6]: For 2nd param expected `int` but got `Tuple[]`.
        return "".join(sum(zip(split_text, separators), ())[:-1])

    def insert_chars_per_word(self, text: str) -> str:
        words, spaces = split_words_on_whitespace(text)
        augmented_words = []

        for word in words:
            augmented_words.append(self.insert_chars(word))

        return rejoin_words_and_whitespace(augmented_words, spaces)

    def augment(self, texts: Union[str, List[str]]) -> List[str]:
        texts = texts if isinstance(texts, list) else [texts]

        if self.granularity == "all":
            return [self.insert_chars(text) for text in texts]

        # otherwise, self.granularity == "word"
        return [self.insert_chars_per_word(text) for text in texts]
