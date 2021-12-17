#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from augly.text.augmenters.utils import (
    detokenize,
    rejoin_words_and_whitespace,
    split_words_on_whitespace,
    tokenize,
)


class UtilsTest(unittest.TestCase):
    def test_tokenize(self) -> None:
        # Includes the apostrophe in "can't" but not in "thinkin'".
        tokens1 = tokenize("Can't stop\nthinkin' about you")
        self.assertEqual(tokens1, ["Can't", "stop", "thinkin", "'", "about", "you"])
        # Differs in the missing newline.
        self.assertEqual(detokenize(tokens1), "Can't stop thinkin' about you")
        # Decimal place needs a leading zero or it isn't handled correctly.
        tokens2 = tokenize("Write it: 0.004% not .004% #grammar")
        self.assertEqual(
            tokens2,
            ["Write", "it", ":", "0.004", "%", "not", ".", "004", "%", "#", "grammar"],
        )
        # No leading zero puts the decimal place with 'not' as a period.
        self.assertEqual(detokenize(tokens2), "Write it: 0.004% not. 004 %#grammar")
        # Dollar signs, parentheses, time. The newline is discarded.
        tokens3 = tokenize("Will work for $$ ($100) until 5:00")
        self.assertEqual(
            tokens3,
            ["Will", "work", "for", "$", "$", "(", "$", "100", ")", "until", "5:00"],
        )
        # Detokenized version has no space between the dollar signs and parenthesis.
        self.assertEqual(detokenize(tokens3), "Will work for $$($100) until 5:00")

    def test_split_words_on_whitespace(self) -> None:
        # Preserves the whitespace from the original text.
        words, whitespace = split_words_on_whitespace(
            "Can't stop\nthinkin'  about\tyou"
        )
        self.assertEqual(words, ["Can't", "stop", "thinkin'", "about", "you"])
        self.assertEqual(whitespace, ["", " ", "\n", "  ", "\t", ""])
        self.assertEqual(
            rejoin_words_and_whitespace(words, whitespace),
            "Can't stop\nthinkin'  about\tyou",
        )


if __name__ == "__main__":
    unittest.main()
