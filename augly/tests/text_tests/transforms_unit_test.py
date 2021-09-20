#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import json
import random
import unittest
from typing import Any, Dict, List

import augly.text as txtaugs
from augly.utils import TEXT_METADATA_PATH


def are_equal_metadata(
    actual_meta: List[Dict[str, Any]], expected_meta: List[Dict[str, Any]]
) -> bool:
    if actual_meta == expected_meta:
        return True

    for actual_dict, expected_dict in zip(actual_meta, expected_meta):
        for (act_k, act_v), (exp_k, exp_v) in zip(
            sorted(actual_dict.items(), key=lambda kv: kv[0]),
            sorted(expected_dict.items(), key=lambda kv: kv[0]),
        ):
            if act_k != exp_k:
                return False

            if act_v == exp_v:
                continue

            """
            Allow relative paths in expected metadata: just check that the end of the
            actual path matches the expected path
            """
            if not (
                isinstance(act_v, str)
                and isinstance(exp_v, str)
                and act_v[-len(exp_v):] == exp_v
            ):
                return False

    return True


class TransformsTextUnitTest(unittest.TestCase):
    def test_import(self) -> None:
        try:
            from augly.text import transforms
        except ImportError:
            self.fail("transforms failed to import")
        self.assertTrue(dir(transforms))

    def setUp(self):
        self.metadata = []
        self.maxDiff = None
        random.seed(123)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        with open(TEXT_METADATA_PATH, "r") as f:
            cls.expected_metadata = json.load(f)

        cls.texts = [
            "The quick brown 'fox' couldn't jump over the green, grassy hill."
        ]
        cls.priority_words = ["green", "grassy", "hill"]

        cls.fairness_texts = [
            "The king and queen have a son named Raj and a daughter named Amanda.",
        ]

    def test_ApplyLambda(self) -> None:
        augmented_apply_lambda = txtaugs.ApplyLambda()(
            self.texts, metadata=self.metadata
        )

        self.assertTrue(augmented_apply_lambda[0] == self.texts[0])
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["apply_lambda"]),
        )

    def test_Compose(self) -> None:
        random.seed(1)
        augmented_compose = txtaugs.Compose(
            [
                txtaugs.OneOf([txtaugs.ReplaceSimilarChars(), txtaugs.SimulateTypos()]),
                txtaugs.InsertPunctuationChars(),
                txtaugs.ReplaceFunFonts(),
            ]
        )(self.texts, metadata=self.metadata)

        self.assertEqual(
            augmented_compose,
            [
                "T... h... e...... u... q... i... c... k...... b... r... o... w... "
                "n...... '... f... o... x... '...... c... o... u... d... n... '...... "
                "t...... j... u... m... p...... o... v... e... f...... t... j... e......"
                " g... r... e... e... n...,...... g... r... a... s... s... y...... h... "
                "i...,... l...."
            ],
        )
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["compose"]),
        )

    def test_GetBaseline(self) -> None:
        augmented_baseline = txtaugs.GetBaseline()(self.texts, metadata=self.metadata)

        self.assertTrue(
            augmented_baseline[0]
            == "The quick brown 'fox' couldn't jump over the green, grassy hill."
        )
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["get_baseline"]),
        )

    def test_InsertPunctuationChars(self) -> None:
        aug_punc_text = txtaugs.InsertPunctuationChars("all", 1.0, False)(
            self.texts, metadata=self.metadata
        )

        # Separator inserted between every character (including spaces/punctuation).
        self.assertEqual(
            aug_punc_text,
            [
                "T?h?e? ?q?u?i?c?k? ?b?r?o?w?n? ?'?f?o?x?'? ?c?o?u?l?d?n?'?t? "
                "?j?u?m?p? ?o?v?e?r? ?t?h?e? ?g?r?e?e?n?,? ?g?r?a?s?s?y? ?h?i?l?l?."
            ],
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["insert_punctuation_chars"]
            ),
        )

    def test_InsertWhitespaceChars(self) -> None:
        aug_whitespace_text = txtaugs.InsertWhitespaceChars("all", 1.0, False)(
            self.texts, metadata=self.metadata
        )

        # Separator inserted between every character (including spaces/punctuation).
        self.assertEqual(
            aug_whitespace_text,
            [
                "T h e   q u i c k   b r o w n   ' f o x '   c o u l d n ' t   "
                "j u m p   o v e r   t h e   g r e e n ,   g r a s s y   h i l l ."
            ],
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["insert_whitespace_chars"]
            ),
        )

    def test_InsertZeroWidthChars(self) -> None:
        aug_unicode_text = txtaugs.InsertZeroWidthChars("all", 1.0, False)(
            self.texts, metadata=self.metadata
        )

        # Separator inserted between every character (including spaces/punctuation).
        # Renders as: "T‌h‌e‌ ‌q‌u‌i‌c‌k‌ ‌b‌r‌o‌w‌n‌ ‌'‌f‌o‌x‌'‌ ‌c‌o‌u‌l‌d‌n‌'‌t‌ ‌j‌u‌m‌p‌ ‌o‌v‌e‌r‌ ‌t‌h‌e‌ ‌g‌r‌e‌e‌n‌,‌ ‌g‌r‌a‌s‌s‌y‌ ‌h‌i‌l‌l‌."
        self.assertEqual(
            aug_unicode_text,
            [
                "T\u200ch\u200ce\u200c \u200cq\u200cu\u200ci\u200cc\u200ck\u200c "
                "\u200cb\u200cr\u200co\u200cw\u200cn\u200c \u200c'\u200cf\u200co"
                "\u200cx\u200c'\u200c \u200cc\u200co\u200cu\u200cl\u200cd\u200cn"
                "\u200c'\u200ct\u200c \u200cj\u200cu\u200cm\u200cp\u200c \u200co"
                "\u200cv\u200ce\u200cr\u200c \u200ct\u200ch\u200ce\u200c \u200cg"
                "\u200cr\u200ce\u200ce\u200cn\u200c,\u200c \u200cg\u200cr\u200ca"
                "\u200cs\u200cs\u200cy\u200c \u200ch\u200ci\u200cl\u200cl\u200c."
            ],
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["insert_zero_width_chars"]
            ),
        )

    def test_ReplaceBidirectional(self) -> None:
        aug_bidirectional_text = txtaugs.ReplaceBidirectional()(
            self.texts, metadata=self.metadata
        )

        # Renders as: "‮.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb kciuq ehT‬"
        self.assertEqual(
            aug_bidirectional_text,
            [
                "\u202e.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb "
                "kciuq ehT\u202c"
            ],
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["replace_bidirectional"]
            ),
        )

    def test_ReplaceFunFonts(self) -> None:
        aug_fun_fonts = txtaugs.ReplaceFunFonts(aug_p=0.8, vary_fonts=False, n=1)(
            self.texts, metadata=self.metadata
        )

        self.assertTrue(
            aug_fun_fonts[0]
            == "𝑻𝒉𝒆 𝒒𝒖𝒊𝒄𝒌 𝒃𝒓𝒐𝒘𝒏 '𝒇𝒐𝒙' 𝒄𝒐𝒖𝒍𝒅𝒏'𝒕 𝒋𝒖𝒎𝒑 𝒐𝒗𝒆𝒓 𝒕𝒉𝒆 𝒈𝒓𝒆𝒆𝒏, 𝒈𝒓𝒂𝒔𝒔𝒚 𝒉𝒊𝒍𝒍."
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["replace_fun_fonts"]
            ),
        )

    def test_ReplaceSimilarChars(self) -> None:
        aug_chars = txtaugs.ReplaceSimilarChars(aug_word_p=0.3, aug_char_p=0.3)(
            self.texts, metadata=self.metadata
        )

        self.assertTrue(
            aug_chars[0]
            == "The quick brown 'fox' coul|)n' t jump ov3r the green, gI2assy hi|_l."
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["replace_similar_chars"]
            ),
        )

    def test_ReplaceSimilarUnicodeChars(self) -> None:
        aug_unicode_chars = txtaugs.ReplaceSimilarUnicodeChars(
            aug_word_p=0.3, aug_char_p=0.3
        )(self.texts, metadata=self.metadata)

        self.assertTrue(
            aug_unicode_chars[0]
            == "The ჹuick brown 'fox' coul₫n' t jump ov∑r the green, Ĝrassy hîll."
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["replace_similar_unicode_chars"]
            ),
        )

    def test_ReplaceUpsideDown(self) -> None:
        aug_upside_down_text = txtaugs.ReplaceUpsideDown()(
            self.texts, metadata=self.metadata
        )

        self.assertTrue(
            aug_upside_down_text[0]
            == "˙llᴉɥ ʎssɐɹɓ 'uǝǝɹɓ ǝɥʇ ɹǝʌo dɯnɾ ʇ,uplnoɔ ,xoɟ, uʍoɹq ʞɔᴉnb ǝɥꞱ"
        )
        self.assertTrue(
            are_equal_metadata(
                self.metadata, self.expected_metadata["replace_upside_down"]
            ),
        )

    def test_ReplaceWords(self) -> None:
        augmented_words = txtaugs.ReplaceWords()(self.texts, metadata=self.metadata)

        self.assertTrue(augmented_words[0] == self.texts[0])
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["replace_words"]),
        )

    def test_SimulateTypos(self) -> None:
        aug_typo_text = txtaugs.SimulateTypos(
            aug_word_p=0.3, aug_char_p=0.3, typo_type="all"
        )(self.texts, metadata=self.metadata)

        self.assertTrue(
            aug_typo_text[0]
            == "Thw qu(ck brown 'fox' co)uldn' t jamp over the green, grassy hill.",
        )
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["simulate_typos"]),
        )

    def test_SplitWords(self) -> None:
        aug_split_words = txtaugs.SplitWords(aug_word_p=0.3)(
            self.texts, metadata=self.metadata
        )

        self.assertTrue(
            aug_split_words[0]
            == "The quick b rown 'fox' couldn' t j ump over the green, gras sy hill."
        )
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["split_words"]),
        )

    def test_SwapGenderedWords(self) -> None:
        augmented_words = txtaugs.SwapGenderedWords()(
            self.fairness_texts, metadata=self.metadata
        )

        self.assertTrue(
            augmented_words[0]
            == "The queen and king have a daughter named Raj and a son named Amanda.",
        )
        self.assertTrue(
            are_equal_metadata(self.metadata, self.expected_metadata["swap_gendered_words"]),
        )


if __name__ == "__main__":
    unittest.main()
