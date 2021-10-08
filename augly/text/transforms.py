#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import augly.text.functional as F
from augly.utils import (
    CONTRACTIONS_MAPPING,
    FUN_FONTS_PATH,
    GENDERED_WORDS_MAPPING,
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
        **kwargs,
    ) -> List[str]:
        """
        @param texts: a string or a list of text documents to be augmented

        @param force: if set to True, the transform will be applied. Otherwise,
            application is determined by the probability set

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param kwargs: any kwargs the user wants to pass in to override the instance
            variables when calling the augmentation

        @returns: the list of augmented text documents
        """
        assert isinstance(
            texts, (str, list)
        ), "Expected types List[str] or str for variable 'texts'"
        assert isinstance(force, bool), "Expected type bool for variable 'force'"

        if not force and random.random() > self.p:
            return texts if isinstance(texts, list) else [texts]

        return self.apply_transform(texts, metadata, **kwargs)

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified in the instance variables from __init__ (or
        overriding kwargs if passed into this function)
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
        aug_function: Optional[Callable[..., List[str]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Apply a user-defined lambda on a list of text documents

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_function: the augmentation function to be applied onto the text
            (should expect a list of text documents as input and return a list of
            text documents). Will override the value given in __init__ if given here

        @param **kwargs: the input attributes to be passed into the augmentation
            function to be applied. Will override the value given in __init__ if given
            here

        @returns: the list of augmented text documents
        """
        lambda_kwargs = deepcopy(self.kwargs)
        lambda_kwargs.update(**lambda_kwargs)
        return F.apply_lambda(
            texts, aug_function or self.aug_function, **lambda_kwargs, metadata=metadata
        )


class ChangeCase(BaseTransform):
    def __init__(
        self,
        granularity: str = "word",
        cadence: float = 1.0,
        case: str = "random",
        seed: Optional[int] = 10,
        p: float = 1.0,
    ):
        """
        @param granularity: 'all' (case of the entire text is changed), 'word' (case of
            random words is changed), or 'char' (case of random chars is changed)

        @param cadence: how frequent (i.e. between this many characters/words) to change
            the case. Must be at least 1.0. Non-integer values are used as an 'average'
            cadence. Not used for granularity 'all'

        @param case: the case to change words to; valid values are 'lower', 'upper',
            'title', or 'random' (in which case every word will be randomly changed to
            one of the 3 cases)

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.granularity = granularity
        self.cadence = cadence
        self.case = case
        self.seed = seed

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        granularity: Optional[str] = None,
        cadence: Optional[float] = None,
        case: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Changes the case (e.g. upper, lower, title) of random chars, words, or the entire
        text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param granularity: 'all' (case of the entire text is changed), 'word' (case of
            random words is changed), or 'char' (case of random chars is changed). Will
            override the value given in __init__ if given here

        @param cadence: how frequent (i.e. between this many characters/words) to change
            the case. Must be at least 1.0. Non-integer values are used as an 'average'
            cadence. Not used for granularity 'all'. Will override the value given in
            __init__ if given here

        @param case: the case to change words to; valid values are 'lower', 'upper',
            'title', or 'random' (in which case every word will be randomly changed to
            one of the 3 cases). Will override the value given in __init__ if given here

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs. Will override the value given in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.change_case(
            texts,
            granularity=granularity or self.granularity,
            cadence=cadence or self.cadence,
            case=case or self.case,
            seed=seed or self.seed,
            metadata=metadata,
        )


class Contractions(BaseTransform):
    def __init__(
        self,
        aug_p: float = 0.3,
        mapping: Optional[Union[str, Dict[str, Any]]] = CONTRACTIONS_MAPPING,
        max_contraction_length: int = 2,
        seed: Optional[int] = 10,
        p: float = 1.0,
    ):
        """
        @param aug_p: the probability that each pair (or longer string) of words will be
            replaced with the corresponding contraction, if there is one in the mapping

        @param mapping: either a dictionary representing the mapping or an iopath uri
            where the mapping is stored

        @param max_contraction_length: the words in each text will be checked for matches
            in the mapping up to this length; i.e. if 'max_contraction_length' is 3 then
            every substring of 2 *and* 3 words will be checked

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_p = aug_p
        self.mapping = mapping
        self.max_contraction_length = max_contraction_length
        self.seed = seed

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        aug_p: Optional[float] = None,
        mapping: Optional[Union[str, Dict[str, Any]]] = None,
        max_contraction_length: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces pairs (or longer strings) of words with contractions given a mapping

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_p: the probability that each pair (or longer string) of words will be
            replaced with the corresponding contraction, if there is one in the mapping.
            Will override the value given in __init__ if given here

        @param mapping: either a dictionary representing the mapping or an iopath uri
            where the mapping is stored. Will override the value given in __init__ if
            given here

        @param max_contraction_length: the words in each text will be checked for matches
            in the mapping up to this length; i.e. if 'max_contraction_length' is 3 then
            every substring of 2 *and* 3 words will be checked. Will override the value
            given in __init__ if given here

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs. Will override the value given in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.contractions(
            texts,
            aug_p=aug_p or self.aug_p,
            mapping=mapping or self.mapping,
            max_contraction_length=max_contraction_length or self.max_contraction_length,
            seed=seed or self.seed,
            metadata=metadata,
        )


class GetBaseline(BaseTransform):
    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
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
        granularity: Optional[str] = None,
        cadence: Optional[float] = None,
        vary_chars: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Inserts punctuation characters in each input text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
            the cadence resets for each word in the text. Will override the value given
            in __init__ if given here

        @param cadence: how frequent (i.e. between this many characters) to insert
            a punctuation character. Must be at least 1.0. Non-integer values are used
            as an 'average' cadence. Will override the value given in __init__ if given
            here

        @param vary_chars: if true, picks a different punctuation char each time one is
            used instead of just one per word/text. Will override the value given in
            __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.insert_punctuation_chars(
            texts,
            granularity=granularity or self.granularity,
            cadence=cadence or self.cadence,
            vary_chars=vary_chars or self.vary_chars,
            metadata=metadata,
        )


class InsertWhitespaceChars(BaseTransform):
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
            a whitespace character. Must be at least 1.0. Non-integer values
            are used as an 'average' cadence

        @param vary_chars: if true, picks a different whitespace char each time
            one is used instead of just one per word/text

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
        granularity: Optional[str] = None,
        cadence: Optional[float] = None,
        vary_chars: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Inserts whitespace characters in each input text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param granularity: 'all' or 'word' -- if 'word', a new char is picked and
            the cadence resets for each word in the text. Will override the value given
            in __init__ if given here

        @param cadence: how frequent (i.e. between this many characters) to insert
            a whitespace character. Must be at least 1.0. Non-integer values
            are used as an 'average' cadence. Will override the value given in __init__
            if given here

        @param vary_chars: if true, picks a different whitespace char each time
            one is used instead of just one per word/text. Will override the value given
            in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.insert_whitespace_chars(
            texts,
            granularity=granularity or self.granularity,
            cadence=cadence or self.cadence,
            vary_chars=vary_chars or self.vary_chars,
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
        granularity: Optional[str] = None,
        cadence: Optional[float] = None,
        vary_chars: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Inserts zero-width characters in each input text

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param granularity: 'all' or 'word' -- if 'word', a new char is picked
            and the cadence resets for each word in the text. Will override the value
            given in __init__ if given here

        @param cadence: how frequent (i.e. between this many characters) to insert
            a zero-width character. Must be at least 1.0. Non-integer values are used
            as an 'average' cadence. Will override the value given in __init__ if given
            here

        @param vary_chars: If true, picks a different zero-width char each time one is
            used instead of just one per word/text. Will override the value given in
            __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.insert_zero_width_chars(
            texts,
            granularity=granularity or self.granularity,
            cadence=cadence or self.cadence,
            vary_chars=vary_chars or self.vary_chars,
            metadata=metadata,
        )


class MergeWords(BaseTransform):
    def __init__(
        self,
        aug_word_p: float = 0.3,
        min_char: int = 2,
        aug_word_min: int = 1,
        aug_word_max: int = 1000,
        n: int = 1,
        priority_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
        @param aug_word_p: probability of words to be augmented

        @param min_char: minimum # of characters in a word to be merged

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
        aug_word_p: Optional[float] = None,
        min_char: Optional[int] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        priority_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Merges words in the text together

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param min_char: minimum # of characters in a word to be merged. Will override
            the value given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.merge_words(
            texts,
            aug_word_p=aug_word_p or self.aug_word_p,
            min_char=min_char or self.min_char,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            priority_words=priority_words or self.priority_words,
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
        granularity: Optional[str] = None,
        split_word: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Reverses each word (or part of the word) in each input text and uses
        bidirectional marks to render the text in its original order. It reverses
        each word separately which keeps the word order even when a line wraps

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param granularity: the level at which the font is applied; this must be
            either 'word' or 'all'. Will override the value given in __init__ if given
            here

        @param split_word: if true and granularity is 'word', reverses only the
            second half of each word. Will override the value given in __init__ if given
            here

        @returns: the list of augmented text documents
        """
        return F.replace_bidirectional(
            texts,
            granularity=granularity or self.granularity,
            split_word=split_word or self.split_word,
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
        aug_p: Optional[float] = None,
        aug_min: Optional[int] = None,
        aug_max: Optional[int] = None,
        granularity: Optional[str] = None,
        vary_fonts: Optional[bool] = None,
        fonts_path: Optional[str] = None,
        n: Optional[int] = None,
        priority_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces words or characters depending on the granularity with fun fonts applied

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_p: probability of words to be augmented. Will override the value given
            in __init__ if given here

        @param aug_min: minimum # of words to be augmented. Will override the value given
            in __init__ if given here

        @param aug_max: maximum # of words to be augmented. Will override the value given
            in __init__ if given here

        @param granularity: the level at which the font is applied; this
            must be be either word, char, or all. Will override the value given in
            __init__ if given here

        @param vary_fonts: whether or not to switch font in each replacement. Will
            override the value given in __init__ if given here

        @param fonts_path: iopath uri where the fonts are stored. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.replace_fun_fonts(
            texts,
            aug_p=aug_p or self.aug_p,
            aug_min=aug_min or self.aug_min,
            aug_max=aug_max or self.aug_max,
            granularity=granularity or self.granularity,
            vary_fonts=vary_fonts or self.vary_fonts,
            fonts_path=fonts_path or self.fonts_path,
            n=n or self.n,
            priority_words=priority_words or self.priority_words,
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
        aug_char_p: Optional[float] = None,
        aug_word_p: Optional[float] = None,
        min_char: Optional[int] = None,
        aug_char_min: Optional[int] = None,
        aug_char_max: Optional[int] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        mapping_path: Optional[str] = None,
        priority_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces letters in each text with similar characters

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_char_p: probability of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param min_char: minimum # of letters in a word for a valid augmentation. Will
            override the value given in __init__ if given here

        @param aug_char_min: minimum # of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_char_max: maximum # of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param mapping_path: iopath uri where the mapping is stored. Will override the
            value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.replace_similar_chars(
            texts,
            aug_char_p=aug_char_p or self.aug_char_p,
            aug_word_p=aug_word_p or self.aug_word_p,
            min_char=min_char or self.min_char,
            aug_char_min=aug_char_min or self.aug_char_min,
            aug_char_max=aug_char_max or self.aug_char_max,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            mapping_path=mapping_path or self.mapping_path,
            priority_words=priority_words or self.priority_words,
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
        aug_char_p: Optional[float] = None,
        aug_word_p: Optional[float] = None,
        min_char: Optional[int] = None,
        aug_char_min: Optional[int] = None,
        aug_char_max: Optional[int] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        mapping_path: Optional[str] = None,
        priority_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces letters in each text with similar unicodes

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_char_p: probability of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param min_char: minimum # of letters in a word for a valid augmentation. Will
            override the value given in __init__ if given here

        @param aug_char_min: minimum # of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_char_max: maximum # of letters to be replaced in each word. Will
            override the value given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param mapping_path: iopath uri where the mapping is stored. Will override the
            value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.replace_similar_unicode_chars(
            texts,
            aug_char_p=aug_char_p or self.aug_char_p,
            aug_word_p=aug_word_p or self.aug_word_p,
            min_char=min_char or self.min_char,
            aug_char_min=aug_char_min or self.aug_char_min,
            aug_char_max=aug_char_max or self.aug_char_max,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            mapping_path=mapping_path or self.mapping_path,
            priority_words=priority_words or self.priority_words,
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
        aug_p: Optional[float] = None,
        aug_min: Optional[int] = None,
        aug_max: Optional[int] = None,
        granularity: Optional[str] = None,
        n: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Flips words in the text upside down depending on the granularity

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_p: probability of words to be augmented. Will override the value given
            in __init__ if given here

        @param aug_min: minimum # of words to be augmented. Will override the value given
            in __init__ if given here

        @param aug_max: maximum # of words to be augmented. Will override the value given
            in __init__ if given here

        @param granularity: the level at which the font is applied;
            this must be be either word, char, or all. Will override the value given in
            __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.replace_upside_down(
            texts,
            aug_p=aug_p or self.aug_p,
            aug_min=aug_min or self.aug_min,
            aug_max=aug_max or self.aug_max,
            granularity=granularity or self.granularity,
            n=n or self.n,
            metadata=metadata,
        )


class ReplaceWords(BaseTransform):
    def __init__(
        self,
        aug_word_p: float = 0.3,
        aug_word_min: int = 1,
        aug_word_max: int = 1000,
        n: int = 1,
        mapping: Optional[Union[str, Dict[str, Any]]] = None,
        priority_words: Optional[List[str]] = None,
        ignore_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
        @param aug_word_p: probability of words to be augmented

        @param aug_word_min: minimum # of words to be augmented

        @param aug_word_max: maximum # of words to be augmented

        @param n: number of augmentations to be performed for each text

        @param mapping: either a dictionary representing the mapping or an iopath uri where
            the mapping is stored

        @param priority_words: list of target words that the augmenter should prioritize
            to augment first

        @param ignore_words: list of words that the augmenter should not augment

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_word_p = aug_word_p
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.mapping = mapping
        self.priority_words = priority_words
        self.ignore_words = ignore_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        aug_word_p: Optional[float] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        mapping: Optional[Union[str, Dict[str, Any]]] = None,
        priority_words: Optional[List[str]] = None,
        ignore_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces words in each text based on a given mapping

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param mapping: either a dictionary representing the mapping or an iopath uri
            where the mapping is stored. Will override the value given in __init__ if
            given here

        @param priority_words: list of target words that the augmenter should prioritize
            to augment first. Will override the value given in __init__ if given here

        @param ignore_words: list of words that the augmenter should not augment. Will
            override the value given in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.replace_words(
            texts,
            aug_word_p=aug_word_p or self.aug_word_p,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            mapping=mapping or self.mapping,
            priority_words=priority_words or self.priority_words,
            ignore_words=ignore_words or self.ignore_words,
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
        typo_type: str = "all",
        misspelling_dict_path: Optional[str] = MISSPELLING_DICTIONARY_PATH,
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

        @param typo_type: the type of typos to apply to the text; valid values are
            "misspelling", "keyboard", "charmix", or "all"

        @param misspelling_dict_path: iopath uri where the misspelling dictionary is
            stored; must be specified if typo_type is "misspelling" or "all", but
            otherwise can be None

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
        self.typo_type = typo_type
        self.misspelling_dict_path = misspelling_dict_path
        self.priority_words = priority_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        aug_char_p: Optional[float] = None,
        aug_word_p: Optional[float] = None,
        min_char: Optional[int] = None,
        aug_char_min: Optional[int] = None,
        aug_char_max: Optional[int] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        typo_type: Optional[str] = None,
        misspelling_dict_path: Optional[str] = None,
        priority_words: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Simulates typos in each text using misspellings, keyboard distance, and swapping.
        You can specify a typo_type: charmix, which does a combination of character-level
        modifications (delete, insert, substitute, & swap); keyboard, which swaps
        characters which those close to each other on the QWERTY keyboard; misspelling,
        which replaces words with misspellings defined in a dictionary file; or all,
        which will apply a random combination of all 4

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_char_p: probability of letters to be replaced in each word;
            This is only applicable for keyboard distance and swapping. Will override the
            value given in __init__ if given here

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param min_char: minimum # of letters in a word for a valid augmentation;
            This is only applicable for keyboard distance and swapping. Will override the
            value given in __init__ if given here

        @param aug_char_min: minimum # of letters to be replaced/swapped in each word;
            This is only applicable for keyboard distance and swapping. Will override the
            value given in __init__ if given here

        @param aug_char_max: maximum # of letters to be replaced/swapped in each word;
            This is only applicable for keyboard distance and swapping. Will override the
            value given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param typo_type: the type of typos to apply to the text; valid values are
            "misspelling", "keyboard", "charmix", or "all". Will override the value given
            in __init__ if given here

        @param misspelling_dict_path: iopath uri where the misspelling dictionary is
            stored; must be specified if typo_type is "misspelling" or "all", but
            otherwise can be None. Will override the value given in __init__ if given
            here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.simulate_typos(
            texts,
            aug_char_p=aug_char_p or self.aug_char_p,
            aug_word_p=aug_word_p or self.aug_word_p,
            min_char=min_char or self.min_char,
            aug_char_min=aug_char_min or self.aug_char_min,
            aug_char_max=aug_char_max or self.aug_char_max,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            typo_type=typo_type or self.typo_type,
            misspelling_dict_path=misspelling_dict_path or self.misspelling_dict_path,
            priority_words=priority_words or self.priority_words,
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
        aug_word_p: Optional[float] = None,
        min_char: Optional[int] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Splits words in the text into subwords

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param min_char: minimum # of characters in a word for a split. Will override the
            value given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @returns: the list of augmented text documents
        """
        return F.split_words(
            texts,
            aug_word_p=aug_word_p or self.aug_word_p,
            min_char=min_char or self.min_char,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            metadata=metadata,
        )


class SwapGenderedWords(BaseTransform):
    def __init__(
        self,
        aug_word_p: float = 0.3,
        aug_word_min: int = 1,
        aug_word_max: int = 1000,
        n: int = 1,
        mapping: Union[str, Dict[str, str]] = GENDERED_WORDS_MAPPING,
        priority_words: Optional[List[str]] = None,
        ignore_words: Optional[List[str]] = None,
        p: float = 1.0,
    ):
        """
        @param aug_word_p: probability of words to be augmented

        @param aug_word_min: minimum # of words to be augmented

        @param aug_word_max: maximum # of words to be augmented

        @param n: number of augmentations to be performed for each text

        @param mapping: a mapping of words from one gender to another; a mapping can be
            supplied either directly as a dict or as a filepath to a json file containing
            the dict

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first

        @param ignore_words: list of words that the augmenter should not augment

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_word_p = aug_word_p
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        self.mapping = mapping
        self.priority_words = priority_words
        self.ignore_words = ignore_words

    def apply_transform(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        aug_word_p: Optional[float] = None,
        aug_word_min: Optional[int] = None,
        aug_word_max: Optional[int] = None,
        n: Optional[int] = None,
        mapping: Optional[str] = None,
        priority_words: Optional[List[str]] = None,
        ignore_words: Optional[List[str]] = None,
        min_char: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Replaces words in each text based on a provided `mapping`, which can either be a
        dict already constructed mapping words from one gender to another or a file path
        to a dict. Note: the logic in this augmentation was originally written by
        Adina Williams and has been used in influential work, e.g.
        https://arxiv.org/pdf/2005.00614.pdf

        @param texts: a string or a list of text documents to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest length, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param aug_word_p: probability of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_min: minimum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param aug_word_max: maximum # of words to be augmented. Will override the value
            given in __init__ if given here

        @param n: number of augmentations to be performed for each text. Will override
            the value given in __init__ if given here

        @param mapping: a mapping of words from one gender to another; a mapping can be
            supplied either directly as a dict or as a filepath to a json file containing
            the dict. Will override the value given in __init__ if given here

        @param priority_words: list of target words that the augmenter should
            prioritize to augment first. Will override the value given in __init__ if
            given here

        @param ignore_words: list of words that the augmenter should not augment. Will
            override the value given in __init__ if given here

        @returns: the list of augmented text documents
        """
        return F.swap_gendered_words(
            texts,
            aug_word_p=aug_word_p or self.aug_word_p,
            aug_word_min=aug_word_min or self.aug_word_min,
            aug_word_max=aug_word_max or self.aug_word_max,
            n=n or self.n,
            mapping=mapping or self.mapping,
            priority_words=priority_words or self.priority_words,
            ignore_words=ignore_words or self.ignore_words,
            metadata=metadata,
        )
