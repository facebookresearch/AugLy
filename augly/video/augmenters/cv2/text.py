#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle
import random
import string
from typing import Any, List, Iterator, Optional, Tuple

import cv2
import numpy as np
from augly.utils import pathmgr
from augly.video.augmenters.cv2 import BaseCV2Augmenter
from PIL import Image, ImageDraw, ImageFont


CV2_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]


class VideoDistractorByText(BaseCV2Augmenter):
    def __init__(
        self,
        text_len: int,
        text_change_nth: Optional[int] = None,
        fonts: Optional[List[Tuple[Any, Optional[str]]]] = None,
        fontscales: Optional[Tuple[float, float]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: Optional[int] = None,
        random_movement: bool = False,
        topleft: Optional[Tuple[float, float]] = None,
        bottomright: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> None:
        assert text_len > 0, "Text length must be greater than zero"
        assert (
            text_change_nth is None or text_change_nth > 0
        ), "`text_change_nth` must be greater than zero"
        assert fonts is None or all(
            (isinstance(font, (str, ImageFont.ImageFont)) or (font in CV2_FONTS))
            and (chars is None or isinstance(chars, str))
            for font, chars in fonts
        ), "Fonts must be either None or a list of tuples of font (cv2 font, PIL ImageFont, or str path to a .ttf file) & chars file (str path or None)"
        assert fontscales is None or (
            fontscales[0] > 0 and fontscales[1] > fontscales[0]
        ), "Fontscale ranges must be greater than zero and the second value must be greater than the first"  # noqa: B950
        assert thickness is None or (
            type(thickness) == int and thickness > 0
        ), "Invalid thickness provided: must be set to None or be an integer greater than zero"  # noqa: B950

        super().__init__(1, random_movement, topleft, bottomright, **kwargs)

        self.texts = self.random_texts(text_len, text_change_nth)
        self.fonts = self.random_fonts(fonts)
        self.fontscales = self.random_fontscales(fontscales)
        self.colors = BaseCV2Augmenter.random_colors(colors)
        self.thickness = self.random_thickness(thickness)

    def random_texts(
        self, text_len: int, text_change_nth: Optional[int]
    ) -> Iterator[List[float]]:
        def random_text(n):
            return [random.random() for _ in range(n)]

        iframe = 0
        if not text_change_nth:
            text = random_text(text_len)
        while True:
            if text_change_nth and iframe % text_change_nth == 0:
                text = random_text(text_len)
            # pyre-fixme[61]: `text` may not be initialized here.
            yield text
            iframe += 1

    def random_fonts(
        self, fonts: Optional[List[Tuple[Any, Optional[str]]]]
    ) -> Iterator[Tuple[Any, List[str]]]:
        fonts_and_chars = fonts or [(font, None) for font in CV2_FONTS]
        while True:
            font_idx = random.randint(0, len(fonts_and_chars) - 1)
            font, chars_path = fonts_and_chars[font_idx]
            if chars_path is not None:
                with pathmgr.open(chars_path, "rb") as f:
                    # pyre-ignore[6]: Expected `typing.IO[bytes]` for 1st positional
                    # only parameter to call `pickle.load` but got
                    # `typing.Union[typing.IO[bytes], typing.IO[str]]`
                    chars = [chr(c) for c in pickle.load(f)]
            else:
                chars = list(string.ascii_letters + string.punctuation)
            yield font, chars

    def random_fontscales(
        self, fontscales: Optional[Tuple[float, float]]
    ) -> Iterator[float]:
        fontscales = fontscales or (2.5, 5)
        while True:
            yield random.uniform(*fontscales)

    def random_thickness(self, thickness: Optional[int]) -> Iterator[int]:
        while True:
            yield thickness or random.randint(2, 5)

    # overrides abstract method of base class
    def apply_augmentation(self, raw_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds text distracts (in various colors, fonts, and positions) to each frame

        @param raw_frame: raw, single RGB/Gray frame

        @returns: the augumented frame
        """
        assert (raw_frame.ndim == 3) and (
            raw_frame.shape[2] == 3
        ), "VideoDistractorByText only accepts RGB images"
        height, width = raw_frame.shape[:2]

        text = next(self.texts)
        font, chars = next(self.fonts)  # pyre-ignore
        fontscale = next(self.fontscales)
        color = next(self.colors)
        thickness = next(self.thickness)
        fraction_x, fraction_y = self.get_origins(0)
        x = int(fraction_x * width)
        y = int(fraction_y * height)
        n = len(chars)
        text_str = "".join([chars[int(c * n)] for c in text])

        distract_frame = raw_frame.copy()
        if isinstance(font, str):
            with pathmgr.open(font, "rb") as f:
                # pyre-fixme[6]: Expected `Union[None,
                #  _typeshed.SupportsRead[bytes], bytes, str]` for 1st param but got
                #  `Union[typing.IO[bytes], typing.IO[str]]`.
                font = ImageFont.truetype(f, int(fontscale * 100))
        if isinstance(
            font,
            (ImageFont.ImageFont, ImageFont.FreeTypeFont, ImageFont.TransposedFont),
        ):
            # To use an ImageFont, we need to convert into PIL
            distract_frame_rgb = cv2.cvtColor(distract_frame, cv2.COLOR_BGR2RGB)
            distract_frame_pil = Image.fromarray(distract_frame_rgb)
            # pyre-fixme[6]: Expected `Optional[ImageFont._Font]` for 3rd param but
            #  got `Union[ImageFont.FreeTypeFont, ImageFont.ImageFont,
            #  ImageFont.TransposedFont]`.
            ImageDraw.Draw(distract_frame_pil).text((x, y), text_str, font=font)
            distract_frame = cv2.cvtColor(np.array(distract_frame_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(
                distract_frame, text_str, (x, y), font, fontscale, color, thickness, cv2.LINE_AA
            )
        return distract_frame
