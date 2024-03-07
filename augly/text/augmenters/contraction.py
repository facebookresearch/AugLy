#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from augly.text.augmenters.utils import detokenize, tokenize
from augly.utils import pathmgr


class ContractionMapping:
    def __init__(self, mapping: Optional[Union[str, Dict[str, Any]]]):
        if isinstance(mapping, str):
            local_mapping_path = pathmgr.get_local_path(mapping)
            with open(local_mapping_path) as json_file:
                self.mapping = {k.lower(): v for k, v in json.load(json_file).items()}
        elif isinstance(mapping, Dict):
            self.mapping = mapping
        else:
            self.mapping = {}

    def replace(self, text: str) -> Optional[str]:
        new_text = self.mapping.get(text.lower(), None)
        if new_text is not None and text[0].isupper():
            new_text = new_text.capitalize()
        return new_text


class ContractionAugmenter:
    """Augmenter that replaces words based on a given mapping"""

    def __init__(
        self,
        aug_p: float,
        mapping: Optional[Union[str, Dict[str, Any]]],
        max_contraction_length: int = 2,
        seed: Optional[int] = 10,
    ):
        assert max_contraction_length >= 2, "Must set 'max_contraction_length' >= 2"
        self.aug_p = aug_p
        self.contraction_mapping = self.get_mapping(mapping)
        self.max_contraction_length = max_contraction_length
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def get_mapping(
        self, mapping: Optional[Union[str, Dict[str, Any]]]
    ) -> ContractionMapping:
        return ContractionMapping(mapping)

    def substitute_contractions(self, text: str) -> str:
        """
        Returns a text where random words are replaced using the specified mapping

        @param text: the text to which the word substitution will be applied
        """
        results = []
        tokens = tokenize(text)

        for c_len in range(2, self.max_contraction_length + 1):
            i = 0
            while i <= len(tokens) - c_len:
                result = tokens[i]
                if self.rng.rand() <= self.aug_p:
                    contraction = self.contraction_mapping.replace(
                        " ".join(tokens[i : i + c_len])
                    )
                    if contraction is not None:
                        result = contraction
                        i += c_len - 1
                results.append(result)
                i += 1

            results.extend(tokens[-c_len + 1 :])

        return detokenize(results)

    def augment(self, texts: Union[str, List[str]]) -> List[str]:
        texts_list = [texts] if isinstance(texts, str) else texts
        return [self.substitute_contractions(text) for text in texts_list]
