#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from augly.text.augmenters.utils import detokenize, tokenize
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
from nlpaug.augmenter.word import WordAugmenter  # @manual
from nlpaug.util import Action  # @manual


class BaselineAugmenter(WordAugmenter):
    """Baseline augmenter that serves as comparison with the original text"""

    def __init__(self):
        super().__init__(action=Action.SUBSTITUTE)

    def substitute(self, data: str) -> str:
        """
        Returns a text that should be the same as the original input text

        @param data: the text that will be tokenized and detokenized
        """
        return detokenize(tokenize(data))
