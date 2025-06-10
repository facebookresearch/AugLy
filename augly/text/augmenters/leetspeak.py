#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from augly.text.augmenters.encode_text_strategy import EncodeTextAugmentation
from augly.text.augmenters.utils import Encoding
from nlpaug.util import Method


class LeetSpeak(EncodeTextAugmentation):
    def __init__(
        self,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        method: Method,
    ):
        super().__init__(
            name="LeetSpeak",
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            encoder=Encoding.LEETSPEAK,
            method=str(method),
        )
        assert 0 <= aug_min <= aug_max
        assert 0 <= aug_p <= 1

    def encode(self, input_string: str) -> str:
        leet_map = {
            "a": ["4", "@"],
            "b": ["8"],
            "e": ["3"],
            "g": ["6"],
            "i": ["1", "!"],
            "l": ["1"],
            "o": ["0"],
            "s": ["5", "$"],
            "t": ["7", "+"],
            "z": ["2"],
        }
        input_string = input_string.lower()
        return "".join(
            random.choice(leet_map.get(char, [char])) for char in input_string
        )
