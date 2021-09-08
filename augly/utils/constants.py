#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import random

from augly.utils.base_paths import (
    AUDIO_ASSETS_DIR,
    EMOJI_DIR,
    FONTS_DIR,
    IMG_MASK_DIR,
    SCREENSHOT_TEMPLATES_DIR,
    TEXT_DIR,
)


# color constants
DEFAULT_COLOR = (0, 0, 0)
WHITE_RGB_COLOR = (255, 255, 255)
RED_RGB_COLOR = (255, 0, 0)

# audio & video rate constants
DEFAULT_FRAME_RATE = 10
DEFAULT_SAMPLE_RATE = 44100

# text constants
DEFAULT_TEXT_INDICES = [random.randrange(1, 1000) for _ in range(5)]

# `overlay_stripes` constants
SUPPORTED_LINE_TYPES = [
    "dotted",
    "dashed",
    "solid"
]

# screenshot augmentation assets
BBOXES_PATH = os.path.join(SCREENSHOT_TEMPLATES_DIR, "bboxes.json")

# audio assets
SILENT_AUDIO_PATH = os.path.join(AUDIO_ASSETS_DIR, "silent.flac")

# image augmentation assets
IMG_MASK_PATH = os.path.join(IMG_MASK_DIR, "dfdc_mask.png")

"""
All emoji assets used in AugLy come from Twemoji: https://twemoji.twitter.com/
Copyright 2020 Twitter, Inc and other contributors.
Code licensed under the MIT License: http://opensource.org/licenses/MIT
Graphics licensed under CC-BY 4.0: https://creativecommons.org/licenses/by/4.0/
"""
SMILEY_EMOJI_DIR = os.path.join(EMOJI_DIR, "smileys")
EMOJI_PATH = os.path.join(SMILEY_EMOJI_DIR, "smiling_face_with_heart_eyes.png")

"""
All font assets used in AugLy come from Google Noto fonts: https://www.google.com/get/noto/
Noto is a trademark of Google Inc. Noto fonts are open source.
All Noto fonts are published under the SIL Open Font License, Version 1.1
"""
FONT_LIST_PATH = os.path.join(FONTS_DIR, "list")
FONT_PATH = os.path.join(FONTS_DIR, "NotoNaskhArabic-Regular.ttf")
MEME_DEFAULT_FONT = os.path.join(FONTS_DIR, "Raleway-ExtraBold.ttf")

# text augmentation assets
FUN_FONTS_PATH = os.path.join(TEXT_DIR, "fun_fonts.json")
FUN_FONTS_GREEK_PATH = os.path.join(TEXT_DIR, "fun_fonts_greek.json")
UNICODE_MAPPING_PATH = os.path.join(TEXT_DIR, "letter_unicode_mapping.json")
MISSPELLING_DICTIONARY_PATH = os.path.join(TEXT_DIR, "misspelling.json")

"""
Text fairness augmentation assets: the feminine & masculine word lists were provided to
us by Adina Williams and are the same ones used in Dinan et al., 2020
(https://arxiv.org/pdf/2005.00614.pdf), which aggregated such word lists from
Zhao et al., 2018b, 2019 (https://aclanthology.org/D18-1521.pdf) and Hoyle et al., 2019
(https://aclanthology.org/P19-1167.pdf). We constructed the gendered words mapping file
from the feminine & masculine word lists for ease of use with the `swap_gendered_words`
augmentation to avoid having to re-compute the mapping
"""
GENDERED_WORDS_MAPPING = os.path.join(TEXT_DIR, "gendered_words_mapping.json")
