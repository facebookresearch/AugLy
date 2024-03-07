#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, List, Union


class TextReplacementAugmenter:
    """
    Replaces the input text entirely with some specified text
    """

    def augment(
        self,
        texts: Union[str, List[str]],
        replace_text: Union[str, Dict[str, str]],
    ) -> Union[str, List[str]]:
        return (
            [self.replace(text, replace_text) for text in texts]
            if isinstance(texts, list)
            else self.replace(texts, replace_text)
        )

    def replace(self, text: str, replace_text: Union[str, Dict[str, str]]) -> str:
        return (
            replace_text.get(text, text)
            if isinstance(replace_text, dict)
            else replace_text
        )
