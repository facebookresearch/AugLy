#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import List, Union

from augly.text.augmenters.encode_text_strategy import EncodeTextAugmentation


class EncodeText:
    def __init__(self, encoder: EncodeTextAugmentation):
        self.encoder = encoder

    def augmenter(self, input_string: Union[List[str], str]) -> Union[List[str], str]:
        return self.encoder.augment(input_string, 1)
