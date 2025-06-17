#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import codecs
from typing import Literal

from augly.text.augmenters.encode_text_strategy import EncodeTextAugmentation


class Base64(EncodeTextAugmentation):
    def __init__(
        self,
        aug_min: int,
        aug_max: int,
        aug_p: float,
        granularity: Literal["all", "word", "char"],
    ):
        super().__init__(
            name="Base64",
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            encoder="base64",
            granularity=granularity,
        )
        assert granularity in {
            "all",
            "word",
        }, f"Base64 only supports granularity type 'all' or 'word', found type {granularity}"
        assert 0 <= aug_min <= aug_max
        assert 0 <= aug_p <= 1

    def encode(self, input_string: str) -> str:
        encoded_bytes = codecs.encode(input_string.encode("utf-8"), "base64")
        return encoded_bytes.decode("utf-8").strip()
