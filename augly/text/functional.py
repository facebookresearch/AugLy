#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import augly.text.augmenters as a
import augly.text.utils as txtutils
from augly.utils import (
    FUN_FONTS_PATH,
    GENDERED_WORDS_MAPPING,
    MISSPELLING_DICTIONARY_PATH,
    UNICODE_MAPPING_PATH,
)


def apply_lambda(
    texts: Union[str, List[str]],
    aug_function: Callable[..., List[str]] = lambda x: x,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[str]:
    """
    Apply a user-defined lambda on a list of text documents

    @param texts: a string or a list of text documents to be augmented

    @param aug_function: the augmentation function to be applied onto the text
        (should expect a list of text documents as input and return a list of
        text documents)

    @param **kwargs: the input attributes to be passed into the augmentation
        function to be applied

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    assert callable(aug_function), (
        repr(type(aug_function).__name__) + " object is not callable"
    )

    func_kwargs = deepcopy(locals())
    if aug_function is not None:
        try:
            func_kwargs["aug_function"] = aug_function.__name__
        except AttributeError:
            func_kwargs["aug_function"] = type(aug_function).__name__
    func_kwargs = txtutils.get_func_kwargs(metadata, func_kwargs)

    aug_texts = aug_function(texts, **kwargs)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="apply_lambda",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def get_baseline(
    texts: Union[str, List[str]],
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Generates a baseline by tokenizing and detokenizing the text

    @param texts: a string or a list of text documents to be augmented

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    baseline_aug = a.BaselineAugmenter()
    aug_texts = baseline_aug.augment(texts, 1)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="get_baseline",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def insert_punctuation_chars(
    texts: Union[str, List[str]],
    granularity: str = "all",
    cadence: float = 1.0,
    vary_chars: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Inserts punctuation characters in each input text

    @param texts: a string or a list of text documents to be augmented

    @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
        the cadence resets for each word in the text

    @param cadence: how frequent (i.e. between this many characters) to insert a
        punctuation character. Must be at least 1.0. Non-integer values are used
        as an 'average' cadence

    @param vary_chars: if true, picks a different punctuation char each time one
        is used instead of just one per word/text

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented texts
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    punctuation_aug = a.InsertionAugmenter(
        "punctuation", granularity, cadence, vary_chars
    )
    aug_texts = punctuation_aug.augment(texts)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="insert_punctuation_chars",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def insert_whitespace_chars(
    texts: Union[str, List[str]],
    granularity: str = "all",
    cadence: float = 1.0,
    vary_chars: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Inserts whitespace characters in each input text

    @param texts: a string or a list of text documents to be augmented

    @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
        the cadence resets for each word in the text

    @param cadence: how frequent (i.e. between this many characters) to insert a
        whitespace character. Must be at least 1.0. Non-integer values are used
        as an 'average' cadence

    @param vary_chars: if true, picks a different whitespace char each time one
        is used instead of just one per word/text

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented texts
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    whitespace_aug = a.InsertionAugmenter(
        "whitespace", granularity, cadence, vary_chars
    )
    aug_texts = whitespace_aug.augment(texts)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="insert_whitespace_chars",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def insert_zero_width_chars(
    texts: Union[str, List[str]],
    granularity: str = "all",
    cadence: float = 1.0,
    vary_chars: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Inserts zero-width characters in each input text

    @param texts: a string or a list of text documents to be augmented

    @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
        the cadence resets for each word in the text

    @param cadence: how frequent (i.e. between this many characters) to insert
        a zero-width character. Must be at least 1.0. Non-integer values are
        used as an 'average' cadence

    @param vary_chars: if true, picks a different zero-width char each time one
        is used instead of just one per word/text

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented texts
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    zero_width_aug = a.InsertionAugmenter(
        "zero_width", granularity, cadence, vary_chars
    )
    aug_texts = zero_width_aug.augment(texts)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="insert_zero_width_chars",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_bidirectional(
    texts: Union[str, List[str]],
    granularity: str = "all",
    split_word: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Reverses each word (or part of the word) in each input text and uses
    bidirectional marks to render the text in its original order. It reverses
    each word separately which keeps the word order even when a line wraps

    @param texts: a string or a list of text documents to be augmented

    @param granularity: the level at which the font is applied; this must be either
        'word' or 'all'

    @param split_word: if true and granularity is 'word', reverses only the second
        half of each word

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented texts
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    bidirectional_aug = a.BidirectionalAugmenter(granularity, split_word)
    aug_texts = bidirectional_aug.augment(texts)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_bidirectional",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_fun_fonts(
    texts: Union[str, List[str]],
    aug_p: float = 0.3,
    aug_min: int = 1,
    aug_max: int = 10000,
    granularity: str = "all",
    vary_fonts: bool = False,
    fonts_path: str = FUN_FONTS_PATH,
    n: int = 1,
    priority_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
):
    """
    Replaces words or characters depending on the granularity with fun fonts applied

    @param texts: a string or a list of text documents to be augmented

    @param aug_p: probability of words to be augmented

    @param aug_min: minimum # of words to be augmented

    @param aug_max: maximum # of words to be augmented

    @param granularity: the level at which the font is applied; this must be be
        either word, char, or all

    @param vary_fonts: whether or not to switch font in each replacement

    @param fonts_path: iopath uri where the fonts are stored

    @param n: number of augmentations to be performed for each text

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    fun_fonts_aug = a.FunFontsAugmenter(
        granularity, aug_min, aug_max, aug_p, vary_fonts, fonts_path, priority_words
    )
    aug_texts = fun_fonts_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_fun_fonts",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_similar_chars(
    texts: Union[str, List[str]],
    aug_char_p: float = 0.3,
    aug_word_p: float = 0.3,
    min_char: int = 2,
    aug_char_min: int = 1,
    aug_char_max: int = 1000,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    mapping_path: Optional[str] = None,
    priority_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Replaces letters in each text with similar characters

    @param texts: a string or a list of text documents to be augmented

    @param aug_char_p: probability of letters to be replaced in each word

    @param aug_word_p: probability of words to be augmented

    @param min_char: minimum # of letters in a word for a valid augmentation

    @param aug_char_min: minimum # of letters to be replaced in each word

    @param aug_char_max: maximum # of letters to be replaced in each word

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param mapping_path: iopath uri where the mapping is stored

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    char_aug = a.LetterReplacementAugmenter(
        min_char,
        aug_char_min,
        aug_char_max,
        aug_char_p,
        aug_word_min,
        aug_word_max,
        aug_word_p,
        mapping_path,
        priority_words,
    )
    aug_texts = char_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_similar_chars",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_similar_unicode_chars(
    texts: Union[str, List[str]],
    aug_char_p: float = 0.3,
    aug_word_p: float = 0.3,
    min_char: int = 2,
    aug_char_min: int = 1,
    aug_char_max: int = 1000,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    mapping_path: str = UNICODE_MAPPING_PATH,
    priority_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Replaces letters in each text with similar unicodes

    @param texts: a string or a list of text documents to be augmented

    @param aug_char_p: probability of letters to be replaced in each word

    @param aug_word_p: probability of words to be augmented

    @param min_char: minimum # of letters in a word for a valid augmentation

    @param aug_char_min: minimum # of letters to be replaced in each word

    @param aug_char_max: maximum # of letters to be replaced in each word

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param mapping_path: iopath uri where the mapping is stored

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    unicode_aug = a.LetterReplacementAugmenter(
        min_char,
        aug_char_min,
        aug_char_max,
        aug_char_p,
        aug_word_min,
        aug_word_max,
        aug_word_p,
        mapping_path,
        priority_words,
    )
    aug_texts = unicode_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_similar_unicode_chars",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_upside_down(
    texts: Union[str, List[str]],
    aug_p: float = 0.3,
    aug_min: int = 1,
    aug_max: int = 1000,
    granularity: str = "all",
    n: int = 1,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Flips words in the text upside down depending on the granularity

    @param texts: a string or a list of text documents to be augmented

    @param aug_p: probability of words to be augmented

    @param aug_min: minimum # of words to be augmented

    @param aug_max: maximum # of words to be augmented

    @param granularity: the level at which the font is applied; this must be
        either word, char, or all

    @param n: number of augmentations to be performed for each text

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    upside_down_aug = a.UpsideDownAugmenter(granularity, aug_min, aug_max, aug_p)
    aug_texts = upside_down_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_upside_down",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def replace_words(
    texts: Union[str, List[str]],
    aug_word_p: float = 0.3,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    mapping: Optional[Union[str, Dict[str, Any]]] = None,
    priority_words: Optional[List[str]] = None,
    ignore_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Replaces words in each text based on a given mapping

    @param texts: a string or a list of text documents to be augmented

    @param aug_word_p: probability of words to be augmented

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param mapping: either a dictionary representing the mapping or an iopath uri where
        the mapping is stored

    @param priority_words: list of target words that the augmenter should prioritize to
        augment first

    @param ignore_words: list of words that the augmenter should not augment

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    word_aug = a.WordReplacementAugmenter(
        aug_word_min, aug_word_max, aug_word_p, mapping, priority_words, ignore_words
    )
    aug_texts = word_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="replace_words",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def simulate_typos(
    texts: Union[str, List[str]],
    aug_char_p: float = 0.3,
    aug_word_p: float = 0.3,
    min_char: int = 2,
    aug_char_min: int = 1,
    aug_char_max: int = 1,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    typo_type: str = "all",
    misspelling_dict_path: Optional[str] = MISSPELLING_DICTIONARY_PATH,
    priority_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Simulates typos in each text using misspellings, keyboard distance, and swapping.
    You can specify a typo_type: charmix, which does a combination of character-level
    modifications (delete, insert, substitute, & swap); keyboard, which swaps characters
    which those close to each other on the QWERTY keyboard; misspelling, which replaces
    words with misspellings defined in a dictionary file; or all, which will apply a
    random combination of all 4

    @param texts: a string or a list of text documents to be augmented

    @param aug_char_p: probability of letters to be replaced in each word;
        This is only applicable for keyboard distance and swapping

    @param aug_word_p: probability of words to be augmented

    @param min_char: minimum # of letters in a word for a valid augmentation;
        This is only applicable for keyboard distance and swapping

    @param aug_char_min: minimum # of letters to be replaced/swapped in each word;
        This is only applicable for keyboard distance and swapping

    @param aug_char_max: maximum # of letters to be replaced/swapped in each word;
        This is only applicable for keyboard distance and swapping

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param typo_type: the type of typos to apply to the text; valid values are
        "misspelling", "keyboard", "charmix", or "all"

    @param misspelling_dict_path: iopath uri where the misspelling dictionary is stored;
        must be specified if typo_type is "misspelling" or "all", but otherwise can be
        None

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    typo_aug = a.TypoAugmenter(
        min_char,
        aug_char_min,
        aug_char_max,
        aug_char_p,
        aug_word_min,
        aug_word_max,
        aug_word_p,
        typo_type,
        misspelling_dict_path,
        priority_words,
    )
    aug_texts = typo_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="simulate_typos",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def split_words(
    texts: Union[str, List[str]],
    aug_word_p: float = 0.3,
    min_char: int = 4,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    priority_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Splits words in the text into subwords

    @param texts: a string or a list of text documents to be augmented

    @param aug_word_p: probability of words to be augmented

    @param min_char: minimum # of characters in a word for a split

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    split_aug = a.SplitWordsAugmenter(
        min_char, aug_word_min, aug_word_max, aug_word_p, priority_words
    )
    aug_texts = split_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="split_words",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts


def swap_gendered_words(
    texts: Union[str, List[str]],
    aug_word_p: float = 0.3,
    aug_word_min: int = 1,
    aug_word_max: int = 1000,
    n: int = 1,
    mapping: Union[str, Dict[str, str]] = GENDERED_WORDS_MAPPING,
    priority_words: Optional[List[str]] = None,
    ignore_words: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Replaces words in each text based on a provided `mapping`, which can either be a dict
    already constructed mapping words from one gender to another or a file path to a
    dict. Note: the logic in this augmentation was originally written by Adina Williams
    and has been used in influential work, e.g. https://arxiv.org/pdf/2005.00614.pdf

    @param texts: a string or a list of text documents to be augmented

    @param aug_word_p: probability of words to be augmented

    @param aug_word_min: minimum # of words to be augmented

    @param aug_word_max: maximum # of words to be augmented

    @param n: number of augmentations to be performed for each text

    @param mapping: a mapping of words from one gender to another; a mapping can be
        supplied either directly as a dict or as a filepath to a json file containing the
        dict

    @param priority_words: list of target words that the augmenter should
        prioritize to augment first

    @param ignore_words: list of words that the augmenter should not augment

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest length, etc. will be appended to
        the inputted list. If set to None, no metadata will be appended or returned

    @returns: the list of augmented text documents
    """
    func_kwargs = txtutils.get_func_kwargs(metadata, locals())

    mapping = txtutils.get_gendered_words_mapping(mapping)

    word_aug = a.WordReplacementAugmenter(
        aug_word_min, aug_word_max, aug_word_p, mapping, priority_words, ignore_words
    )
    aug_texts = word_aug.augment(texts, n)

    txtutils.get_metadata(
        metadata=metadata,
        function_name="swap_gendered_words",
        aug_texts=aug_texts,
        **func_kwargs,
    )

    return aug_texts
