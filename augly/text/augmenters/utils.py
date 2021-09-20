#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import re
import regex
from typing import List, Optional, Tuple

from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug import Augmenter  # @manual
from nlpaug.util import Method  # @manual


LANGUAGES = {
    "ar": "ar_AR",
    "de": "de_DE",
    "el": "el_GR",
    "en": "en_XX",
    "es": "es_XX",
    "fr": "fr_XX",
    "id": "id_ID",
    "it": "it_IT",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "pt": "pt_XX",
    "ro": "ro_RO",
    "sv": "sv_SE",
    "th": "th_TH",
    "tr": "tr_TR",
    "vi": "vi_VN",
}

LETTER_CHAR_MAPPING = {
    "a": ["@", "4"],
    "b": ["|3", "13", "!3"],
    "c": ["(", "[", "{", "<"],
    "d": ["|)"],
    "e": ["3"],
    "g": ["6", "9"],
    "h": ["|-|", "/-/", ")-("],
    "i": ["1", "7", "!", "|"],
    "k": ["|<", "1<", "|(", "|{"],
    "l": ["1", "7", "!", "|", "|_"],
    "m": ["|V|", "IVI", "^^", "nn"],
    "n": ["^"],
    "o": ["D", "0", "()", "[]", "<>"],
    "r": ["12", "I2", "|2", "/2"],
    "s": ["$", "5", "z", "Z"],
    "t": ["7", "+"],
    "w": ["vv", "VV", "UU", "uu"],
    "x": ["><", ")("],
    "z": ["2"],
}

WORDNET_LANGUAGE_MAPPING = {
    "ar": "arb",
    "bg": "bul",
    "ca": "cat",
    "da": "dan",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "eu": "eus",
    "fa": "fas",
    "fi": "fin",
    "fr": "fra",
    "gl": "glg",
    "he": "heb",
    "hr": "hrv",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ms": "zsm",
    "nl": "nld",
    "no": "nno",
    "pl": "pol",
    "pt": "por",
    "sl": "slv",
    "sv": "swe",
    "th": "tha",
    "zh": "cmn",
}

WORDNET_POS_MAPPING = {
    "NN": ["n"],
    "NNS": ["n"],
    "NNP": ["n"],
    "NNPS": ["n"],
    "NP": ["n"],
    "VB": ["v"],
    "VBD": ["v"],
    "VBG": ["v"],
    "VBN": ["v"],
    "VBZ": ["v"],
    "VBP": ["v"],
    "JJ": ["a", "s"],
    "JJR": ["a", "s"],
    "JJS": ["a", "s"],
    "IN": ["a", "s"],
    "RB": ["r"],
    "RBR": ["r"],
    "RBS": ["r"],
}

NEGATE_CONTRACTIONS = {
    "aren't": "are",
    "can't": "can",
    "couldn't": "could",
    "didn't": "did",
    "doesn't": "does",
    "hadn't": "had",
    "haven't": "have",
    "mustn't": "must",
    "oughtn't": "ought",
    "shouldn't": "should",
    "shouldn't've": "should've",
    "wasn't": "was",
    "weren't": "were",
    "won't": "will",
    "wouldn't": "would",
}

PARENS_BRACKETS = [
    (re.compile(r"\s([\[\(\{\<])\s"), r" \1"),
    (re.compile(r"\s([\]\)\}\>])\s"), r"\1 "),
]

PUNCTUATION = [
    (re.compile(r"\s([-])\s"), r"\1"),  # Zero pad
    (re.compile(r"(\s)?([#])\s"), r"\2"),  # Hashtags
    (re.compile(r"\s([,;:%])\s"), r"\1 "),  # Right pad
    (re.compile(r"([\$])\s([\d])"), r"\1\2"),  # $ amounts
    (re.compile(r"([\$])\s"), r"\1"),  # Consecutive $ signs
    (re.compile(r"(\s)?([\.\?\!])"), r"\2"),  # End punctuation
]

QUOTES = [
    (re.compile(r"([\'])\s(.*?)\s([\'])"), r"\1\2\3"),
    (re.compile(r"([\"])\s(.*?)\s([\"])"), r"\1\2\3"),
    (re.compile(r"\s(\')\s"), r"\1 "),
]

SPLIT_BY_WHITESPACE = re.compile(r"(\S+)")

TOKENIZER_REGEXPS = (
    r"""
    (?: [\w]+['][\w]+)             # Contractions
    |
    (?:[+\-]?\d+[,/.:^]\d+)        # Numbers (fractions, decimals, time, ratios)
    |
    (?:[\w_]+)                     # Words without punctuation
    |
    (?:\S)                         # Everything else
    """,
)

TOKENIZER_REGEX = regex.compile(
    r"""(%s)""" % "|".join(TOKENIZER_REGEXPS), regex.VERBOSE | regex.UNICODE
)

UPSIDE_DOWN_CHAR_MAPPING = dict(
    zip(
        "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA0987654321&_?!\"'.,;",
        "zʎxʍʌnʇsɹbdouɯlʞɾᴉɥɓɟǝpɔqɐZ⅄XMΛՈꞱSᴚტԀONW⅂ꓘᒋIH⅁ℲƎᗡƆᗺⱯ068ㄥ9Ϛ߈Ɛᘔ⇂⅋‾¿¡„,˙'؛",
    )
)


def tokenize(text: str) -> List[str]:
    return TOKENIZER_REGEX.findall(text)


def detokenize(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = " " + text + " "

    for regexp, substitution in PARENS_BRACKETS:
        text = regexp.sub(substitution, text)

    for regexp, substitution in PUNCTUATION:
        text = regexp.sub(substitution, text)

    for regexp, substitution in QUOTES:
        text = regexp.sub(substitution, text)

    return text.strip()


def split_words_on_whitespace(text: str) -> Tuple[List[str], List[str]]:
    # Beginning and end are treated as whitespace even if they are empty strings
    split_elements = SPLIT_BY_WHITESPACE.split(text)
    # Return words and whitespace separately
    return split_elements[1::2], split_elements[::2]


def rejoin_words_and_whitespace(words: List[str], whitespace: List[str]) -> str:
    # The split regex returns one whitespace element than word
    assert len(whitespace) == len(words) + 1, "Input lengths do not match!"
    # Add a dummy entry to 'words' so we can zip it easily, then drop it
    ordered_elements = sum(zip(whitespace, words + [""]), ())[:-1]
    return "".join(ordered_elements)


def validate_augmenter_params(
    aug_char_min: int,
    aug_char_max: int,
    aug_char_p: float,
    aug_word_min: int,
    aug_word_max: int,
    aug_word_p: float,
) -> None:
    assert aug_char_min >= 0, "aug_char_min must be non-negative"
    assert aug_char_max >= 0, "aug_char_max must be non-negative"
    assert 0 <= aug_char_p <= 1, "aug_char_p must be a value in the range [0, 1]"
    assert aug_word_min >= 0, "aug_word_min must be non-negative"
    assert aug_word_max >= 0, "aug_word_max must be non-negative"
    assert 0 <= aug_word_p <= 1, "aug_word_p must be a value in the range [0,1]"


def get_aug_idxes(
    augmenter: Augmenter,
    tokens: List[str],
    filtered_idxes: List[int],
    aug_cnt: int,
    mode: str,
    min_char: Optional[int] = None,
) -> List[int]:
    assert (
        mode in Method.getall()
    ), "Expected 'mode' to be a value defined in nlpaug.util.method.Method"

    priority_idxes = []
    priority_words = getattr(augmenter, "priority_words", None)
    ignore_words = getattr(augmenter, "ignore_words", set())

    if mode == Method.WORD and priority_words is not None:
        priority_words_set = set(priority_words)
        for i, token in enumerate(tokens):
            if token in priority_words_set and token not in ignore_words:
                if min_char is None or len(token) >= min_char:
                    priority_idxes.append(i)

    idxes = []
    for i in filtered_idxes:
        if i not in priority_idxes:
            if (
                min_char is None
                or len(tokens[i]) >= min_char
                and tokens[i] not in ignore_words
            ):
                idxes.append(i)

    if len(priority_idxes) + len(idxes) == 0:
        return []

    if len(priority_idxes) <= aug_cnt:
        aug_idxes = priority_idxes
        aug_cnt -= len(priority_idxes)
        if len(idxes) < aug_cnt:
            aug_cnt = len(idxes)
        aug_idxes += augmenter.sample(idxes, aug_cnt)
    else:
        aug_idxes = augmenter.sample(priority_idxes, aug_cnt)

    return aug_idxes
