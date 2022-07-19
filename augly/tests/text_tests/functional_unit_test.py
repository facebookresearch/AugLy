#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every UTF8

import random
import unittest

from augly import text as txtaugs
from augly.utils import FUN_FONTS_GREEK_PATH


class FunctionalTextUnitTest(unittest.TestCase):
    def test_import(self) -> None:
        try:
            from augly.text import functional
        except ImportError:
            self.fail("functional failed to import")
        self.assertTrue(dir(functional))

    def setUp(self):
        random.seed(123)
        self.texts = [
            "The quick brown 'fox' couldn't jump over the green, grassy hill.",
        ]
        self.priority_words = ["green", "grassy", "hill"]

        self.fairness_texts = [
            "The king and queen have a son named Raj and a daughter named Amanda.",
        ]

    def test_apply_lambda(self) -> None:
        augmented_apply_lambda = txtaugs.apply_lambda(self.texts)
        self.assertTrue(augmented_apply_lambda[0] == self.texts[0])

    def test_change_case(self) -> None:
        augmented_words = txtaugs.change_case(self.texts[0], cadence=3.0, case="upper")
        self.assertTrue(
            augmented_words[0]
            == "THE quick brown 'FOX' couldn't jump OVER the green, GRASSY hill.",
        )

    def test_contractions(self) -> None:
        augmented_words = txtaugs.contractions(
            "I would call him but I do not know where he has gone", aug_p=0.7
        )
        self.assertTrue(
            augmented_words[0] == "I would call him but I don't know where he's gone"
        )

    def test_get_baseline(self) -> None:
        augmented_baseline = txtaugs.get_baseline(self.texts)
        self.assertTrue(
            augmented_baseline[0]
            == "The quick brown 'fox' couldn't jump over the green, grassy hill."
        )

    def test_insert_punctuation_chars(self) -> None:
        augmented_every_char = txtaugs.insert_punctuation_chars(
            self.texts, "all", 1.0, False
        )
        # Separator inserted between every character (including spaces/punctuation).
        self.assertEqual(
            augmented_every_char,
            [
                "T.h.e. .q.u.i.c.k. .b.r.o.w.n. .'.f.o.x.'. .c.o.u.l.d.n.'.t. "
                ".j.u.m.p. .o.v.e.r. .t.h.e. .g.r.e.e.n.,. .g.r.a.s.s.y. .h.i.l.l.."
            ],
        )
        augmented_per_word = txtaugs.insert_punctuation_chars(
            self.texts, "word", 1.0, False
        )
        # Each word uses a different separator; no separators around whitespace.
        self.assertEqual(
            augmented_per_word,
            [
                "T;h;e q?u?i?c?k b-r-o-w-n ';f;o;x;' c?o?u?l?d?n?'?t j.u.m.p o-v-e-r "
                "t...h...e g...r...e...e...n..., g:r:a:s:s:y h:i:l:l:."
            ],
        )
        augmented_wider_cadence = txtaugs.insert_punctuation_chars(
            self.texts, "all", 2.7, False
        )
        # Separators are every 2-3 (avg. 2.7) characters.
        self.assertEqual(
            augmented_wider_cadence,
            [
                "The. qu.ick. b.row.n '.fo.x' .cou.ld.n't. ju.mp .ov.er .the. g.ree.n, "
                ".gr.ass.y h.ill.."
            ],
        )
        augmented_varying_char = txtaugs.insert_punctuation_chars(
            self.texts, "all", 2.0, True
        )
        # Each separator is chosen independently.
        self.assertEqual(
            augmented_varying_char,
            [
                "Th?e ,qu,ic!k .br,ow.n :'f.ox!' ;co...ul.dn?'t' j.um...p :ov!er' "
                "t-he, g're?en:, ;gr'as!sy, h-il;l."
            ],
        )

    def test_insert_whitespace_chars(self) -> None:
        augmented_every_char = txtaugs.insert_whitespace_chars(
            self.texts, "all", 1.0, False
        )
        # Separator inserted between every character (including spaces/punctuation).
        self.assertEqual(
            augmented_every_char,
            [
                "T h e   q u i c k   b r o w n   ' f o x '   c o u l d n ' t   "
                "j u m p   o v e r   t h e   g r e e n ,   g r a s s y   h i l l ."
            ],
        )
        augmented_per_word = txtaugs.insert_whitespace_chars(
            self.texts, "word", 1.0, False
        )
        # Each word uses a different separator; no separators around whitespace.
        self.assertEqual(
            augmented_per_word,
            [
                "T\nh\ne q u i c k b\rr\ro\rw\rn '\nf\no\nx\n' c o u l d n ' t "
                "j u m p o\rv\re\rr t\x0bh\x0be g\x0br\x0be\x0be\x0bn\x0b, "
                "g\nr\na\ns\ns\ny h\ni\nl\nl\n."
            ],
        )
        augmented_wider_cadence = txtaugs.insert_whitespace_chars(
            self.texts, "all", 2.7, False
        )
        # Separators are every 2-3 (avg. 2.7) characters.
        self.assertEqual(
            augmented_wider_cadence,
            [
                "The  qu ick  b row n ' fo x'  cou ld n't  ju mp  ov er  the  "
                "g ree n,  gr ass y h ill ."
            ],
        )
        augmented_varying_char = txtaugs.insert_whitespace_chars(
            self.texts, "all", 2.0, True
        )
        # Each separator is chosen independently.
        self.assertEqual(
            augmented_varying_char,
            [
                "Th e \nqu\nic\tk  br\now n \r'f ox\t' \nco\x0cul dn 't\x0b "
                "j um\x0cp \rov\ter\x0c t\x0bhe\n g\x0bre\ten\r, "
                "\rgr\x0bas\tsy\n h\x0bil\rl."
            ],
        )

    def test_insert_text(self) -> None:
        # Single insertion in random location
        insert_single_word = txtaugs.insert_text(self.texts, ["wolf", "sheep"], seed=42)
        self.assertEqual(
            insert_single_word,
            ["wolf The quick brown 'fox' couldn't jump over the green, grassy hill."],
        )

        # Three insertions in random locations
        insert_multiple = txtaugs.insert_text(
            self.texts, ["wolf", "sheep"], num_insertions=3
        )
        self.assertEqual(
            insert_multiple,
            [
                "The quick brown wolf 'fox' couldn't jump wolf over the sheep green, grassy hill."
            ],
        )

        # Single insertion in prepend mode
        prepend = txtaugs.insert_text(
            self.texts, ["wolf", "sheep"], insertion_location="prepend"
        )
        self.assertEqual(
            prepend,
            ["wolf The quick brown 'fox' couldn't jump over the green, grassy hill."],
        )

        append = txtaugs.insert_text(
            self.texts, ["wolf", "sheep"], insertion_location="append"
        )
        # Single insertion in append mode
        self.assertEqual(
            append,
            ["The quick brown 'fox' couldn't jump over the green, grassy hill. wolf"],
        )

    def test_insert_zero_width_chars(self) -> None:
        augmented_every_char = txtaugs.insert_zero_width_chars(
            self.texts, "all", 1.0, False
        )
        # Separator inserted between every character (including spaces/punctuation).
        # Renders as: "T‌h‌e‌ ‌q‌u‌i‌c‌k‌ ‌b‌r‌o‌w‌n‌ ‌'‌f‌o‌x‌'‌ ‌c‌o‌u‌l‌d‌n‌'‌t‌ ‌j‌u‌m‌p‌ ‌o‌v‌e‌r‌ ‌t‌h‌e‌ ‌g‌r‌e‌e‌n‌,‌ ‌g‌r‌a‌s‌s‌y‌ ‌h‌i‌l‌l‌."
        self.assertEqual(
            augmented_every_char,
            [
                "T\u200bh\u200be\u200b \u200bq\u200bu\u200bi\u200bc\u200bk\u200b "
                "\u200bb\u200br\u200bo\u200bw\u200bn\u200b \u200b'\u200bf\u200bo"
                "\u200bx\u200b'\u200b \u200bc\u200bo\u200bu\u200bl\u200bd\u200bn"
                "\u200b'\u200bt\u200b \u200bj\u200bu\u200bm\u200bp\u200b \u200bo"
                "\u200bv\u200be\u200br\u200b \u200bt\u200bh\u200be\u200b \u200bg"
                "\u200br\u200be\u200be\u200bn\u200b,\u200b \u200bg\u200br\u200ba"
                "\u200bs\u200bs\u200by\u200b \u200bh\u200bi\u200bl\u200bl\u200b."
            ],
        )
        augmented_per_word = txtaugs.insert_zero_width_chars(
            self.texts, "word", 1.0, False
        )
        # Each word uses a different separator; no separators around whitespace.
        # Renders as: "T⁡h⁡e q‌u‌i‌c‌k b⁣r⁣o⁣w⁣n '⁡f⁡o⁡x⁡' c‌o‌u‌l‌d‌n‌'‌t j​u​m​p o⁣v⁣e⁣r t⁢h⁢e g⁢r⁢e⁢e⁢n⁢, g​r​a​s​s​y h‍i‍l‍l‍."
        self.assertEqual(
            augmented_per_word,
            [
                "T\u2061h\u2061e q\u200cu\u200ci\u200cc\u200ck b\u2063r\u2063o"
                "\u2063w\u2063n '\u2061f\u2061o\u2061x\u2061' c\u200co\u200cu"
                "\u200cl\u200cd\u200cn\u200c'\u200ct j\u200bu\u200bm\u200bp o"
                "\u2063v\u2063e\u2063r t\u2062h\u2062e g\u2062r\u2062e\u2062e"
                "\u2062n\u2062, g\u200br\u200ba\u200bs\u200bs\u200by h\u200di"
                "\u200dl\u200dl\u200d."
            ],
        )
        augmented_wider_cadence = txtaugs.insert_zero_width_chars(
            self.texts, "all", 2.7, False
        )
        # Separators are every 2-3 (avg. 2.7) characters.
        # Renders as: "The‍ qu‍ick‍ b‍row‍n '‍fo‍x' ‍cou‍ld‍n't‍ ju‍mp ‍ov‍er ‍the‍ g‍ree‍n, ‍gr‍ass‍y h‍ill‍."
        self.assertEqual(
            augmented_wider_cadence,
            [
                "The\u200d qu\u200dick\u200d b\u200drow\u200dn '\u200dfo\u200dx"
                "' \u200dcou\u200dld\u200dn't\u200d ju\u200dmp \u200dov\u200der"
                " \u200dthe\u200d g\u200dree\u200dn, \u200dgr\u200dass\u200dy h"
                "\u200dill\u200d."
            ],
        )
        augmented_varying_char = txtaugs.insert_zero_width_chars(
            self.texts, "all", 2.0, True
        )
        # Each separator is chosen independently.
        # Renders as: "Th‍e ‍qu‌ic​k ⁠br​ow⁡n ​'f‍ox⁠' ⁤co​ul‌dn⁣'t​ j⁤um⁡p ‍ov⁣er⁣ t‍he⁣ g‌re⁡en⁡, ⁣gr‍as⁠sy⁣ h⁡il⁢l."
        self.assertEqual(
            augmented_varying_char,
            [
                "Th\u200de \u200dqu\u200cic\u200bk \u2060br\u200bow\u2061n \u200b"
                "'f\u200dox\u2060' \u2064co\u200bul\u200cdn\u2063't\u200b j\u2064u"
                "m\u2061p \u200dov\u2063er\u2063 t\u200dhe\u2063 g\u200cre\u2061e"
                "n\u2061, \u2063gr\u200das\u2060sy\u2063 h\u2061il\u2062l."
            ],
        )

    def test_merge_words(self) -> None:
        augmented_split_words = txtaugs.merge_words(self.texts, aug_word_p=0.3, n=1)
        self.assertTrue(
            augmented_split_words[0]
            == "Thequick brown 'fox' couldn'tjump overthe green, grassy hill."
        )
        augmented_split_words_targetted = txtaugs.merge_words(
            self.texts, aug_word_p=0.3, n=1, priority_words=self.priority_words
        )
        self.assertTrue(
            augmented_split_words_targetted[0]
            == "The quick brown 'fox' couldn'tjump over the green, grassyhill."
        )

    def test_replace_bidirectional(self) -> None:
        augmented_bidirectional = txtaugs.replace_bidirectional(self.texts)
        # Renders as: "‮.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb kciuq ehT‬"
        self.assertEqual(
            augmented_bidirectional,
            [
                "\u202e.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb kciuq ehT\u202c"
            ],
        )
        augmented_bidirectional_word = txtaugs.replace_bidirectional(
            self.texts, granularity="word"
        )
        # Renders as: "‭‮ehT‬ ‮kciuq‬ ‮nworb‬ ‮'xof'‬ ‮t'ndluoc‬ ‮pmuj‬ ‮revo‬ ‮eht‬ ‮,neerg‬ ‮yssarg‬ ‮.llih‬"
        self.assertEqual(
            augmented_bidirectional_word,
            [
                "\u202d\u202eehT\u202c \u202ekciuq\u202c \u202enworb\u202c \u202e'"
                "xof'\u202c \u202et'ndluoc\u202c \u202epmuj\u202c \u202erevo\u202c"
                " \u202eeht\u202c \u202e,neerg\u202c \u202eyssarg\u202c \u202e.lli"
                "h\u202c"
            ],
        )
        augmented_bidirectional_split = txtaugs.replace_bidirectional(
            self.texts, granularity="word", split_word=True
        )
        # Renders as: "‭T‮eh‬ qu‮kci‬ br‮nwo‬ 'f‮'xo‬ coul‮t'nd‬ ju‮pm‬ ov‮re‬ t‮eh‬ gre‮,ne‬ gra‮yss‬ hi‮.ll‬"
        self.assertEqual(
            augmented_bidirectional_split,
            [
                "\u202dT\u202eeh\u202c qu\u202ekci\u202c br\u202enwo\u202c 'f\u202e"
                "'xo\u202c coul\u202et'nd\u202c ju\u202epm\u202c ov\u202ere\u202c"
                " t\u202eeh\u202c gre\u202e,ne\u202c gra\u202eyss\u202c hi\u202e"
                ".ll\u202c"
            ],
        )

    def test_replace_fun_fonts(self) -> None:
        augmented_fun_fonts_word = txtaugs.replace_fun_fonts(
            self.texts, granularity="word", aug_p=0.3, vary_fonts=False, n=1
        )
        self.assertTrue(
            augmented_fun_fonts_word[0]
            == "The 𝙦𝙪𝙞𝙘𝙠 brown '𝙛𝙤𝙭' 𝙘𝙤𝙪𝙡𝙙𝙣'𝙩 jump over the green, 𝙜𝙧𝙖𝙨𝙨𝙮 hill."
        )
        augmented_fun_fonts_char = txtaugs.replace_fun_fonts(
            self.texts, granularity="char", aug_p=0.3, vary_fonts=True, n=1
        )
        self.assertTrue(
            augmented_fun_fonts_char[0]
            == "T̷he̳ 𝒒uiᴄk 𝙗r𝓸wn 'fo̲x' coul͎dn't jump over t̶h̷e green, 𝑔ra͎ss̳𝒚 ʜill."
        )
        augmented_fun_fonts_all = txtaugs.replace_fun_fonts(
            self.texts, granularity="all", aug_p=1.0, vary_fonts=False, n=1
        )
        self.assertTrue(
            augmented_fun_fonts_all[0]
            == "𝕋𝕙𝕖 𝕢𝕦𝕚𝕔𝕜 𝕓𝕣𝕠𝕨𝕟 '𝕗𝕠𝕩' 𝕔𝕠𝕦𝕝𝕕𝕟'𝕥 𝕛𝕦𝕞𝕡 𝕠𝕧𝕖𝕣 𝕥𝕙𝕖 𝕘𝕣𝕖𝕖𝕟, 𝕘𝕣𝕒𝕤𝕤𝕪 𝕙𝕚𝕝𝕝."
        )
        augmented_fun_fonts_word_targetted = txtaugs.replace_fun_fonts(
            self.texts,
            granularity="word",
            aug_p=0.3,
            vary_fonts=True,
            n=1,
            priority_words=self.priority_words,
        )
        self.assertTrue(
            augmented_fun_fonts_word_targetted[0]
            == "T͓̽h͓̽e͓̽ quick brown 'fox' couldn't jump over the 𝘨𝘳𝘦𝘦𝘯, g̳r̳a̳s̳s̳y̳ h̴i̴l̴l̴."
        )
        augmented_fun_fonts_greek = txtaugs.replace_fun_fonts(
            [
                "Η γρήγορη καφέ αλεπού δεν μπορούσε να πηδήξει πάνω από τον καταπράσινο λόφο."
            ],
            granularity="word",
            aug_p=0.3,
            vary_fonts=True,
            fonts_path=FUN_FONTS_GREEK_PATH,
            n=1.0,
        )
        self.assertTrue(
            augmented_fun_fonts_greek[0]
            == "𝝜 γρήγορη καφέ αλεπού 𝛿𝜀𝜈 μπορούσε να πηδήξει πάνω από 𝞽𝞸𝞶 𝝹𝝰𝞃𝝰𝝿𝞀ά𝞂𝝸𝝼𝝾 λόφο."
        )

    def test_replace_similar_chars(self) -> None:
        augmented_chars = txtaugs.replace_similar_chars(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2
        )
        self.assertEqual(
            augmented_chars,
            [
                "T/-/e quick brown 'fox' coul|)n't jump over the green, grassy hill.",
                "T)-(e quick br0wn 'fox' couldn't jump over the green, g12assy hill.",
            ],
        )
        augmented_chars_targetted = txtaugs.replace_similar_chars(
            self.texts[0],
            aug_word_p=0.3,
            aug_char_p=0.3,
            n=2,
            priority_words=self.priority_words,
        )
        self.assertEqual(
            augmented_chars_targetted,
            [
                "The quic|{ brown 'fox' couldn't jump Dver the green, gr4ssy hill.",
                "7he quick brown 'fox' couldn't jump over the green, gr4ssy hill.",
            ],
        )

    def test_replace_similar_unicode_chars(self) -> None:
        augmented_unicode_chars = txtaugs.replace_similar_unicode_chars(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2
        )
        self.assertEqual(
            augmented_unicode_chars,
            [
                "TĦe ℚuick brown 'fox' coul₫n't jump over the green, grassy hill.",
                "Ŧhe quick browŅ 'fox' couldn't jÙmp over the green, grassy hill.",
            ],
        )
        augmented_unicode_chars_targetted = txtaugs.replace_similar_unicode_chars(
            self.texts[0],
            aug_word_p=0.3,
            aug_char_p=0.3,
            n=2,
            priority_words=self.priority_words,
        )
        self.assertEqual(
            augmented_unicode_chars_targetted,
            [
                "⍡he quick brown 'fox' couldn't jump oveℛ the green, ġrassy hill.",
                "The quick brown 'fox' couldn't jump over thė green, gℝassy hill.",
            ],
        )

    def test_replace_text(self) -> None:
        texts = [
            "The quick brown 'fox' couldn't jump over the green, grassy hill.",
            "The quick brown",
            "jump over the green",
        ]
        replace_texts = {
            "jump over the blue": "jump over the red",
            "The quick brown": "The slow green",
            "couldn't jump": "jumped",
        }

        replace_string = "The slow green"

        augmented_text_from_list = txtaugs.replace_text(texts, replace_texts)
        self.assertEqual(
            augmented_text_from_list,
            [texts[0], replace_texts[texts[1]], texts[2]],
        )

        augmented_text_from_string = txtaugs.replace_text(texts, replace_string)
        self.assertEqual(
            augmented_text_from_string,
            [replace_string, replace_string, replace_string],
        )

        augmented_string_from_list = txtaugs.replace_text(texts[0], replace_texts)
        self.assertTrue(augmented_string_from_list == texts[0])

        augmented_string_from_list = txtaugs.replace_text(texts[1], replace_texts)
        self.assertTrue(augmented_string_from_list == replace_texts[texts[1]])

        augmented_string_from_string = txtaugs.replace_text(texts[2], replace_string)
        self.assertTrue(augmented_string_from_string == replace_string)

    def test_replace_upside_down(self) -> None:
        augmented_upside_down_all = txtaugs.replace_upside_down(self.texts)
        self.assertTrue(
            augmented_upside_down_all[0]
            == "˙llᴉɥ ʎssɐɹɓ 'uǝǝɹɓ ǝɥʇ ɹǝʌo dɯnɾ ʇ,uplnoɔ ,xoɟ, uʍoɹq ʞɔᴉnb ǝɥꞱ"
        )
        augmented_upside_down_words = txtaugs.replace_upside_down(
            self.texts, granularity="word", aug_p=0.3
        )
        self.assertTrue(
            augmented_upside_down_words[0]
            == "ǝɥꞱ ʞɔᴉnb brown 'xoɟ' ʇ,uplnoɔ jump over the green, grassy hill."
        )
        augmented_upside_down_chars = txtaugs.replace_upside_down(
            self.texts[0], granularity="char", aug_p=0.3, n=2
        )
        self.assertEqual(
            augmented_upside_down_chars,
            [
                "Thǝ buiɔk qrown 'fox, couldn,t jump over ʇɥe green' gɹassʎ hill.",
                "Ʇɥǝ qnᴉɔk qɹown 'fox' conldn,t jnɯp over the gɹeen, grassy ɥᴉll ˙",
            ],
        )

    def test_replace_words(self) -> None:
        augmented_words = txtaugs.replace_words(self.texts, aug_word_p=0.3)
        self.assertTrue(augmented_words[0] == self.texts[0])

        augmented_words = txtaugs.replace_words(
            self.texts,
            mapping={"jump": "hop", "brown": "orange", "green": "blue", "the": "a"},
            aug_word_p=1.0,
        )
        self.assertTrue(
            augmented_words[0]
            == "A quick orange 'fox' couldn't hop over a blue, grassy hill.",
        )

        augmented_words = txtaugs.replace_words(
            self.texts,
            mapping={"jump": "hop", "brown": "orange", "green": "blue", "the": "a"},
            aug_word_p=1.0,
            ignore_words=["green", "jump"],
        )
        self.assertTrue(
            augmented_words[0]
            == "A quick orange 'fox' couldn't jump over a green, grassy hill.",
        )

    def test_simulate_typos(self) -> None:
        augmented_typos = txtaugs.simulate_typos(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2, typo_type="misspelling"
        )
        self.assertEqual(
            augmented_typos,
            [
                "Ther quick brown 'fox' couldn' t jump over the green, grassy hill.",
                "Teh quick brown 'fox' couldn' t jump over tghe green, grassy hill.",
            ],
        )

        augmented_typos_targetted = txtaugs.simulate_typos(
            self.texts[0],
            aug_word_p=0.3,
            n=2,
            priority_words=self.priority_words,
            typo_type="charmix",
        )
        self.assertEqual(
            augmented_typos_targetted,
            [
                "The quick buown 'fox' couldn' t jump over he rgeen, rgassy lhill.",
                "The quick brown 'fox' couldn' t nump o^er the gre$n, grasys ill.",
            ],
        )

    def test_split_words(self) -> None:
        augmented_split_words = txtaugs.split_words(self.texts[0], aug_word_p=0.3, n=2)
        self.assertEqual(
            augmented_split_words,
            [
                "The qui ck brown 'fox' c ouldn't j ump over the green, grassy hi ll.",
                "The qu ick bro wn 'fox' could n't jump over the gre en, grassy hill.",
            ],
        )
        augmented_split_words_targetted = txtaugs.split_words(
            self.texts[0], aug_word_p=0.3, n=2, priority_words=self.priority_words
        )
        self.assertEqual(
            augmented_split_words_targetted,
            [
                "The quick br own 'fox' couldn't jump over the g reen, gras sy h ill.",
                "The quick brown 'fox' couldn't jump ov er the g reen, g rassy hi ll.",
            ],
        )

    def test_swap_gendered_words(self) -> None:
        augmented_gender_swap_words = txtaugs.swap_gendered_words(
            self.fairness_texts[0], aug_word_p=0.3
        )
        self.assertTrue(
            augmented_gender_swap_words
            == "The queen and king have a daughter named Raj and a son named Amanda.",
        )

        ignore_augmented_gender_swap_words = txtaugs.swap_gendered_words(
            self.fairness_texts[0], aug_word_p=0.3, ignore_words=["son"]
        )
        self.assertTrue(
            ignore_augmented_gender_swap_words
            == "The queen and king have a son named Raj and a son named Amanda.",
        )


if __name__ == "__main__":
    unittest.main()
