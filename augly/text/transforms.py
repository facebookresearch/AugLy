#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Any, Callable, Dict, List, Optional, Union

import augly.text.functional as F
from augly.utils import (
    FUN_FONTS_PATH,
    MISSPELLING_DICTIONARY_PATH,
    UNICODE_MAPPING_PATH,
)


"""
Base Classes for Transforms
"""


class BaseTransform(object):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(
        self,
        texts: Union[str, List[str]],
        force: bool = False,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        @param texts: a string or a list of text documents to be augmented

        @param force: if set to True, the transform will be applied. Otherwise,
            application is determined by the probability set

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        assert isinstance(
            texts, (str, list)
        ), "Expected types List[str] or str for variable 'texts'"
        assert isinstance(force, bool), "Expected type bool for variable 'force'"

        if not force and random.random() > self.p:
            return texts if isinstance(texts, list) else [texts]

        return self.apply_transform(texts, metadata)

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()


"""
Non-Random Transforms

These classes below are essentially class-based versions of the augmentation
functions previously defined. These classes were developed such that they can
be used with Composition operators (such as `torchvision`'s) and to support
use cases where a specific transform with specific attributes needs to be
applied multiple times.

Example:
 >>> texts = ["hello world", "bye planet"]
 >>> tsfm = InsertPunctuationChars(granularity="all", p=0.5)
 >>> aug_texts = tsfm(texts)
"""


class ApplyLambda(BaseTransform):
    def __init__(
        self,
        aug_function: Callable[..., List[str]] = lambda x: x,
        p: float = 1.0,
        **kwargs,
    ):
        """
        @param aug_function: the augmentation function to be applied onto the text
            (should expect a list of text documents as input and return a list of
            text documents)

        @param p: the probability of the transform being applied; default value is 1.0

        @param **kwargs: the input attributes to be passed into the augmentation
            function to be applied
        """
        super().__init__(p)
        self.aug_function = aug_function
        self.kwargs = kwargs

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Apply a user-defined lambda on a list of text documents

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.apply_lambda(
            texts, self.aug_function, **self.kwargs, metadata=metadata
        )


class GetBaseline(BaseTransform):
    def apply_transform(
        self,
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
        return F.get_baseline(texts, metadata=metadata)


class InsertPunctuationChars(BaseTransform):
    def __init__(
        self,
        granularity: str = "all",
        cadence: float = 1.0,
        vary_chars: bool = False,
        p: float = 1.0,
    ):
        """
        @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
            the cadence resets for each word in the text

        @param cadence: how frequent (i.e. between this many characters) to insert
            a punctuation character. Must be at least 1.0. Non-integer values are used
            as an 'average' cadence

        @param vary_chars: if true, picks a different punctuation char each time one is
            used instead of just one per word/text

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.granularity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Inserts punctuation characters in each input text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.insert_punctuation_chars(
            texts,
            granularity=self.granularity,
            cadence=self.cadence,
            vary_chars=self.vary_chars,
            metadata=metadata,
        )


class InsertZeroWidthChars(BaseTransform):
    def __init__(
        self,
        granularity: str = "all",
        cadence: float = 1.0,
        vary_chars: bool = False,
        p: float = 1.0,
    ):
        """
        @param granularity: 'all' or 'word' -- if 'word', a new char is picked
            and the cadence resets for each word in the text

        @param cadence: how frequent (i.e. between this many characters) to insert
            a zero-width character. Must be at least 1.0. Non-integer values are used
            as an 'average' cadence

        @param vary_chars: If true, picks a different zero-width char each time one is
            used instead of just one per word/text

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.granularity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Inserts zero-width characters in each input text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.insert_zero_width_chars(
            texts,
            granularity=self.granularity,
            cadence=self.cadence,
            vary_chars=self.vary_chars,
            metadata=metadata,
        )


class ReplaceBidirectional(BaseTransform):
    def __init__(
        self,
        granularity: str = "all",
        split_word: bool = False,
        p: float = 1.0,
    ):
        """
        @param granularity: the level at which the font is applied; this must be
            either 'word' or 'all'

        @param split_word: if true and granularity is 'word', reverses only the
            second half of each word

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.granularity = granularity
        self.split_word = split_word

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Reverses each word (or part of the word) in each input text and uses
        bidirectional marks to render the text in its original order. It reverses
        each word separately which keeps the word order even when a line wraps

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.replace_bidirectional(
            texts,
            granularity=self.granularity,
            split_word=self.split_word,
            metadata=metadata,
        )


class ReplaceFunFonts(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        aug_min: int = 1,
        aug_max: int = 10000,
        granularity: str = "all",
        vary_fonts: bool = False,
        fonts_path: str = FUN_FONTS_PATH,
        n: int = 1,
        priority_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
        @param aug_p: probability of words to be augmented

        @param aug_min: minimum # of words to be augmented

        @param aug_max: maximum # of words to be augmented

        @param granularity: the level at which the font is applied; this
            must be be either word, char, or all

        @param vary_fonts: whether or not to switch font in each replacement

        @param fonts_path: iopath uri where the fonts are stored

        @param n: number of augmentations to be performed for each text

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.vary_fonts = vary_fonts
        self.fonts_path = fonts_path
        self.n = n
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Replaces words or characters depending on the granularity with fun fonts applied

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.replace_fun_fonts(
            texts,
            aug_p=self.aug_p,
            aug_min=self.aug_min,
            aug_max=self.aug_max,
            granularity=self.granularity,
            vary_fonts=self.vary_fonts,
            fonts_path=self.fonts_path,
            n=self.n,
            priority_words=self.priority_words,
            metadata=metadata,
        )


class ReplaceSimilarChars(BaseTransform):
    def __init__(
        self,
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
        p: float = 1.0,
    ):
        """
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

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.mapping_path = mapping_path
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Replaces letters in each text with similar characters

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.replace_similar_chars(
            texts,
            aug_char_p=self.aug_char_p,
            aug_word_p=self.aug_word_p,
            min_char=self.min_char,
            aug_char_min=self.aug_char_min,
            aug_char_max=self.aug_char_max,
            aug_word_min=self.aug_word_min,
            aug_word_max=self.aug_word_max,
            n=self.n,
            mapping_path=self.mapping_path,
            priority_words=self.priority_words,
            metadata=metadata,
        )


class ReplaceSimilarUnicodeChars(BaseTransform):
    def __init__(
        self,
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
        p: float = 1.0,
    ):
        """
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

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.mapping_path = mapping_path
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Replaces letters in each text with similar unicodes

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.replace_similar_unicode_chars(
            texts,
            aug_char_p=self.aug_char_p,
            aug_word_p=self.aug_word_p,
            min_char=self.min_char,
            aug_char_min=self.aug_char_min,
            aug_char_max=self.aug_char_max,
            aug_word_min=self.aug_word_min,
            aug_word_max=self.aug_word_max,
            n=self.n,
            mapping_path=self.mapping_path,
            priority_words=self.priority_words,
            metadata=metadata,
        )


class ReplaceUpsideDown(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        aug_min: int = 1,
        aug_max: int = 1000,
        granularity: str = "all",
        n: int = 1,
        p: float = 1.0,
    ):
        """
        @param aug_p: probability of words to be augmented

        @param aug_min: minimum # of words to be augmented

        @param aug_max: maximum # of words to be augmented

        @param granularity: the level at which the font is applied;
            this must be be either word, char, or all

        @param n: number of augmentations to be performed for each text

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.n = n

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Flips words in the text upside down depending on the granularity

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.replace_upside_down(
            texts,
            aug_p=self.aug_p,
            aug_min=self.aug_min,
            aug_max=self.aug_max,
            granularity=self.granularity,
            n=self.n,
            metadata=metadata,
        )


class SimulateTypos(BaseTransform):
    def __init__(
        self,
        aug_char_p: float = 0.3,
        aug_word_p: float = 0.3,
        min_char: int = 2,
        aug_char_min: int = 1,
        aug_char_max: int = 1,
        aug_word_min: int = 1,
        aug_word_max: int = 1000,
        n: int = 1,
        misspelling_dict_path: str = MISSPELLING_DICTIONARY_PATH,
        priority_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
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

        @param misspelling_dict_path: iopath uri where the misspelling dictionary is stored

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.misspelling_dict_path = misspelling_dict_path
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Simulates typos in each text using misspellings, keyboard distance, and swapping

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.simulate_typos(
            texts,
            aug_char_p=self.aug_char_p,
            aug_word_p=self.aug_word_p,
            min_char=self.min_char,
            aug_char_min=self.aug_char_min,
            aug_char_max=self.aug_char_max,
            aug_word_min=self.aug_word_min,
            aug_word_max=self.aug_word_max,
            n=self.n,
            misspelling_dict_path=self.misspelling_dict_path,
            priority_words=self.priority_words,
            metadata=metadata,
        )


class SplitWords(BaseTransform):
    def __init__(
        self,
        aug_word_p: float = 0.3,
        min_char: int = 4,
        aug_word_min: int = 1,
        aug_word_max: int = 1000,
        n: int = 1,
        priority_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
        @param aug_word_p: probability of words to be augmented

        @param min_char: minimum # of characters in a word for a split

        @param aug_word_min: minimum # of words to be augmented

        @param aug_word_max: maximum # of words to be augmented

        @param n: number of augmentations to be performed for each text

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Splits words in the text into subwords

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @returns: the list of augmented text documents
        """
        return F.split_words(
            texts,
            aug_word_p=self.aug_word_p,
            min_char=self.min_char,
            aug_word_min=self.aug_word_min,
            aug_word_max=self.aug_word_max,
            n=self.n,
            priority_words=self.priority_words,
            metadata=metadata,
        )
