#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from random import Random
from typing import List, Optional, Union


class InsertTextAugmenter:
    """
    Inserts some specified text into the input text a given number of times at a given
    location
    """

    def __init__(
        self,
        num_insertions: int = 1,
        insertion_location: str = "random",
        seed: Optional[int] = 10,
    ):
        VALID_LOCATIONS = {"prepend", "append", "random"}
        assert (
            insertion_location in VALID_LOCATIONS
        ), f"{insertion_location} has to be one of {VALID_LOCATIONS}"
        assert num_insertions >= 1, "num_insertions has to be a positive number"

        self.num_insertions = num_insertions
        self.insertion_location = insertion_location
        self.seeded_random = Random(seed)

    def augment(
        self,
        texts: Union[str, List[str]],
        insert_text: List[str],
    ) -> List[str]:
        transformed_texts = []
        for text in texts:
            transformed_text = text
            for _ in range(self.num_insertions):
                text_choice = self.seeded_random.choice(insert_text)

                if self.insertion_location == "append":
                    transformed_text = f"{transformed_text} {text_choice}"
                elif self.insertion_location == "prepend":
                    transformed_text = f"{text_choice} {transformed_text}"
                else:
                    words = transformed_text.split()
                    random_loc = self.seeded_random.randint(0, len(words))
                    words.insert(random_loc, text_choice)
                    transformed_text = " ".join(words)
            transformed_texts.append(transformed_text)

        return transformed_texts
