#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Dict, List, Optional, Union


def apply_lambda_intensity(aug_function: str, **kwargs) -> float:
    intensity_func = globals().get(f"{aug_function}_intensity")
    return intensity_func(**kwargs) if intensity_func else 100.0


def change_case_intensity(granularity: str, cadence: float, **kwargs) -> float:
    return char_insertion_intensity_helper(granularity, cadence)


def contractions_intensity(aug_p: float, **kwargs) -> float:
    return aug_p * 100.0


def get_baseline_intensity(**kwargs) -> float:
    # get_baseline simply tokenizes and detokenizes text and at most adds extra spaces
    return 0.0


def insert_punctuation_chars_intensity(
    granularity: str, cadence: float, **kwargs
) -> float:
    return char_insertion_intensity_helper(granularity, cadence)


def insert_text_intensity(num_insertions: int, **kwargs) -> float:
    assert (
        isinstance(num_insertions, int) and num_insertions >= 0
    ), "Expected 'num_insertions' to be a nonnegative integer"

    max_insertions = 10
    return min((num_insertions / max_insertions) * 100.0, 100.0)


def insert_whitespace_chars_intensity(
    granularity: str, cadence: float, **kwargs
) -> float:
    return char_insertion_intensity_helper(granularity, cadence)


def insert_zero_width_chars_intensity(
    granularity: str, cadence: float, **kwargs
) -> float:
    return char_insertion_intensity_helper(granularity, cadence)


def merge_words_intensity(aug_word_p: float, aug_word_max: int, **kwargs) -> float:
    return replace_intensity_helper(aug_word_p, aug_word_max)


def replace_bidirectional_intensity(**kwargs) -> float:
    return 100.0


def replace_fun_fonts_intensity(
    aug_p: float, aug_max: int, granularity: str, **kwargs
) -> float:
    return 100.0 if granularity == "all" else replace_intensity_helper(aug_p, aug_max)


def replace_similar_chars_intensity(
    aug_char_p: float, aug_word_p: float, aug_char_max: int, aug_word_max: int, **kwargs
) -> float:
    # we only care if aug_*_max is zero or not, so it's okay to multiply the values here
    return replace_intensity_helper(
        aug_word_p * aug_char_p, aug_word_max * aug_char_max
    )


def replace_similar_unicode_chars_intensity(
    aug_char_p: float, aug_word_p: float, aug_char_max: int, aug_word_max: int, **kwargs
) -> float:
    # we only care if aug_*_max is zero or not, so it's okay to multiply the values here
    return replace_intensity_helper(
        aug_word_p * aug_char_p, aug_word_max * aug_char_max
    )


def replace_text_intensity(
    texts: Union[List[str], str], replace_text: Union[Dict[str, str], str], **kwargs
) -> float:
    return (
        100.0
        if isinstance(replace_text, str) or any(t in texts for t in replace_text)
        else 0.0
    )


def replace_upside_down_intensity(
    aug_p: float, aug_max: int, granularity: str, **kwargs
) -> float:
    return 100.0 if granularity == "all" else replace_intensity_helper(aug_p, aug_max)


def replace_words_intensity(
    aug_word_p: float,
    aug_word_max: int,
    mapping: Optional[Union[str, Dict[str, Any]]],
    **kwargs,
) -> float:
    return 0.0 if not mapping else replace_intensity_helper(aug_word_p, aug_word_max)


def simulate_typos_intensity(
    aug_char_p: float, aug_word_p: float, aug_char_max: int, aug_word_max: int, **kwargs
) -> float:
    # we only care if aug_*_max is zero or not, so it's okay to multiply the values here
    return replace_intensity_helper(
        aug_word_p * aug_char_p, aug_word_max * aug_char_max
    )


def split_words_intensity(aug_word_p: float, aug_word_max: int, **kwargs) -> float:
    return replace_intensity_helper(aug_word_p, aug_word_max)


def swap_gendered_words_intensity(
    aug_word_p: float,
    aug_word_max: int,
    **kwargs,
) -> float:
    return replace_intensity_helper(aug_word_p, aug_word_max)


def char_insertion_intensity_helper(granularity: str, cadence: float) -> float:
    return 100.0 if granularity == "all" else (1 / cadence) * 100.0


def replace_intensity_helper(aug_p: float, aug_max: int) -> float:
    return 0.0 if aug_max == 0 else aug_p * 100.0
