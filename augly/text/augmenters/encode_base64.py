#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import codecs

from augly.text.augmenters.utils import detokenize, get_aug_idxes, tokenize
from nlpaug.augmenter.word import Augmenter  # @manual
from nlpaug.util import Action, Method  # @manual


class EncodeBase64(Augmenter):
    def __init__(self, granularity="all", aug_min=1, aug_max=10, aug_p=0.3):

        assert granularity in ["char", "word", "all"]
        assert 0 <= aug_min <= aug_max
        assert 0 <= aug_p <= 1

        self.granularity = granularity
        super().__init__(
            name="EncodeBase64",
            action=Action.SUBSTITUTE,
            method=Method.WORD,
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
        )

    def clean(self, data):

        if isinstance(data, list):
            return [self.clean(d) for d in data]
        elif isinstance(data, str):
            return data
        elif data is None:
            return ""
        else:
            return str(data)

    def encode_text(self, input_string: str) -> str:
        if not isinstance(input_string, str):
            raise TypeError("Input must be a string")

        encoded_bytes = codecs.encode(input_string.encode("utf-8"), "base64")
        return encoded_bytes.decode("utf-8").strip()

    def substitute(self, data) -> str:
        if self.granularity == "all":
            return self.encode_text(data)

        tokens = tokenize(data)
        if not tokens:
            return ""

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
                if i in aug_word_idxes:
                    tokens[i] = self.encode_text(token)

        elif self.granularity == "char":
            for t_i, token in enumerate(tokens):
                chars = list(token)
                aug_char_cnt = self._generate_aug_cnt(
                    len(chars), self.aug_min, self.aug_max, self.aug_p
                )
                aug_char_idxes = set(
                    get_aug_idxes(
                        self, chars, list(range(len(chars))), aug_char_cnt, Method.CHAR
                    )
                )
                for c_i, char in enumerate(chars):
                    if c_i in aug_char_idxes:
                        chars[c_i] = self.encode_text(char)
                tokens[t_i] = "".join(chars)
        return detokenize(tokens)
