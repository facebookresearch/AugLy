#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
import unittest

import augly.text as txtaugs


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
            "The quick brown 'fox' couldn't jump over the green, grassy hill."
        ]
        self.priority_words = ["green", "grassy", "hill"]

    def test_apply_lambda(self) -> None:
        augmented_apply_lambda = txtaugs.apply_lambda(self.texts)
        self.assertTrue(augmented_apply_lambda[0] == self.texts[0])

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

    def test_replace_bidirectional(self) -> None:
        augmented_bidirectional = txtaugs.replace_bidirectional(self.texts)
        # Renders as: "‮.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb kciuq ehT‬"
        self.assertEqual(
            augmented_bidirectional,
            [
                "\u202e.llih yssarg ,neerg eht revo pmuj t'ndluoc 'xof' nworb kciuq ehT\u202c"  # noqa: B950
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
            == "ᴛhe̷ 𝒒uic̾k 𝙗r𝓸wn 'fo̶x' coul̴dn'̴t jump over t̳h̴e green, g͓̽ra̷ss̾y̷ h͓̽ill."
        )
        augmented_fun_fonts_all = txtaugs.replace_fun_fonts(
            self.texts, granularity="all", aug_p=1.0, vary_fonts=False, n=1
        )
        self.assertTrue(
            augmented_fun_fonts_all[0]
            == "𝐓𝐡𝐞 𝐪𝐮𝐢𝐜𝐤 𝐛𝐫𝐨𝐰𝐧 '𝐟𝐨𝐱' 𝐜𝐨𝐮𝐥𝐝𝐧'𝐭 𝐣𝐮𝐦𝐩 𝐨𝐯𝐞𝐫 𝐭𝐡𝐞 𝐠𝐫𝐞𝐞𝐧, 𝐠𝐫𝐚𝐬𝐬𝐲 𝐡𝐢𝐥𝐥."
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
            == "The quick brown 'fox' c̲o̲u̲l̲d̲n̲'̲t̲ jump over the g̴r̴e̴e̴n̴, 𝙜𝙧𝙖𝙨𝙨𝙮 h͎i͎l͎l͎."
        )

    def test_replace_similar_chars(self) -> None:
        augmented_chars = txtaugs.replace_similar_chars(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2
        )
        self.assertTrue(
            augmented_chars[0]
            == "7he qui{k brDwn 'fox' c0uldn' t jump ov3r the green, grassy hill."
        )
        self.assertTrue(
            augmented_chars[1]
            == "+he quick |3rown 'f[]x' couldn' t jump Dver 7he green, grassy hill."
        )
        augmented_chars_targetted = txtaugs.replace_similar_chars(
            self.texts[0],
            aug_word_p=0.3,
            aug_char_p=0.3,
            n=2,
            priority_words=self.priority_words,
        )
        self.assertTrue(
            augmented_chars_targetted[0]
            == "The quick 13rown 'fox' couldn' t jump ove/2 the gr3en, 9rassy hi|_l."
        )
        self.assertTrue(
            augmented_chars_targetted[1]
            == "The quic|( brown 'fox' couldn' t jump over +he gree^, grassy hil!."
        )

    def test_replace_similar_unicode_chars(self) -> None:
        augmented_unicode_chars = txtaugs.replace_similar_unicode_chars(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2
        )
        self.assertTrue(
            augmented_unicode_chars[0]
            == "†he qui⊂k browŅ 'fox' coỦldn' t jump o∨er the green, grassy hill."
        )
        self.assertTrue(
            augmented_unicode_chars[1]
            == "The ჹuick brown 'ⓕox' couldn' t jumρ over the ġreen, grassy hilŀ."
        )
        augmented_unicode_chars_targetted = txtaugs.replace_similar_unicode_chars(
            self.texts[0],
            aug_word_p=0.3,
            aug_char_p=0.3,
            n=2,
            priority_words=self.priority_words,
        )
        self.assertTrue(
            augmented_unicode_chars_targetted[0]
            == "Thė quick brown 'fox' couldn' t jump over ₸he grèen, gras$y hiḽl."
        )
        self.assertTrue(
            augmented_unicode_chars_targetted[1]
            == "TΉe quick brown 'fox' couldn' t jump oveř the gree⋒, grasకy hiℒl."
        )

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
        self.assertTrue(
            augmented_upside_down_chars[0]
            == "Thǝ buiɔk qrown 'fox, couldn,t jump over ʇɥe green' gɹassʎ hill."
        )
        self.assertTrue(
            augmented_upside_down_chars[1]
            == "Ʇɥǝ qnᴉɔk qɹown 'fox' conldn,t jnɯp over the gɹeen, grassy ɥᴉll ˙"
        )

    def test_simulate_typos(self) -> None:
        augmented_typos = txtaugs.simulate_typos(
            self.texts[0], aug_word_p=0.3, aug_char_p=0.3, n=2
        )
        self.assertTrue(
            augmented_typos[0]
            == "Tje uqick brown 'fox' oculdn' t ,ump over the green, rgassy hill."
        )
        self.assertTrue(
            augmented_typos[1]
            == "Teh uqick borwn 'fox' couldn' t jump over the grene, grassy hill."
        )
        augmented_typos_targetted = txtaugs.simulate_typos(
            self.texts[0], aug_word_p=0.3, n=2, priority_words=self.priority_words
        )
        self.assertTrue(
            augmented_typos_targetted[0]
            == "The qucik brown 'fox' couldn' t jump ovre the rgeen, geassy hil?."
        )
        self.assertTrue(
            augmented_typos_targetted[1]
            == "The quick brown 'fox' couldn' t jump ovee hten ggeen, grzssy hill."
        )

    def test_split_words(self) -> None:
        augmented_split_words = txtaugs.split_words(self.texts[0], aug_word_p=0.3, n=2)
        self.assertTrue(
            augmented_split_words[0]
            == "The qui ck brown 'fox' c ouldn't j ump over the green, grassy hi ll."
        )
        self.assertTrue(
            augmented_split_words[1]
            == "The qu ick bro wn 'fox' could n't jump over the gre en, grassy hill."
        )
        augmented_split_words_targetted = txtaugs.split_words(
            self.texts[0], aug_word_p=0.3, n=2, priority_words=self.priority_words
        )
        self.assertTrue(
            augmented_split_words_targetted[0]
            == "The quick br own 'fox' couldn't jump over the g reen, gras sy h ill."
        )
        self.assertTrue(
            augmented_split_words_targetted[1]
            == "The quick brown 'fox' couldn't jump ov er the g reen, g rassy hi ll."
        )


if __name__ == "__main__":
    unittest.main()
