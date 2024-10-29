#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageFont


def aug_np_wrapper(
    image: np.ndarray, aug_function: Callable[..., None], **kwargs
) -> np.ndarray:
    """
    This function is a wrapper on all image augmentation functions
    such that a numpy array could be passed in as input instead of providing
    the path to the image or a PIL Image

    @param image: the numpy array representing the image to be augmented

    @param aug_function: the augmentation function to be applied onto the image

    @param **kwargs: the input attributes to be passed into the augmentation function
    """
    pil_image = Image.fromarray(image)
    aug_image = aug_function(pil_image, **kwargs)
    return np.array(aug_image)


def fit_text_in_bbox(
    text: str,
    img_height: int,
    img_width: int,
    font_path: str,
    font_size: int,
    min_font_size: int,
    rand: random.Random,
) -> Tuple[int, int, List[str], int, ImageFont.FreeTypeFont]:
    """Fits text into a bounding box by adjusting font size and x-coordinate

    @param text: Text to fit into bounding box

    @param img_height: Height of image

    @param img_width: Width of image

    @param font_path: Path to font file

    @param font_size: Font size to start with

    @param min_font_size: Minimum font size to try

    @param rand: Random number generator

    @returns: x and y coordinates to start writing, text split into lines, line heigh, and font style
    """
    x_min = int(img_width * 0.05)  # reserves 5% on the left
    x_max = int(img_width * 0.5)  # starts writing at the center of the image
    random_x = rand.randint(
        x_min, x_max
    )  # generate random x-coordinate to start writing

    max_img_width = int(img_width * 0.95)  # reserves 5% on the right side of image

    while True:
        # loads font
        font = ImageFont.truetype(font_path, font_size)

        # wrap text around image
        lines = wrap_text_for_image_overlay(text, font, int(max_img_width - random_x))
        _, _, _, line_height = font.getbbox("hg")

        y_min = int(img_height * 0.05)  # reserves 5% on the top
        y_max = int(img_height * 0.9)  # reseves 10% to the bottom
        y_max -= (
            len(lines) * line_height
        )  # adjust max y-coordinate for text height and number of lines

        if y_max < y_min:
            if random_x > x_min:
                # adjust x-coordinate by 10% to try to fit text
                random_x = int(max(random_x - 0.1 * max_img_width, x_min))

            elif font_size > min_font_size:
                # reduces font size by 1pt to try to fit text
                font_size -= 1
            else:
                raise ValueError("Text too long to fit onto image!")
        else:
            random_y = rand.randint(
                y_min, y_max
            )  # generate random y-coordinate to start writing
            return random_x, random_y, lines, line_height, font


def wrap_text_for_image_overlay(
    text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> List[str]:
    """Wraps text around an image

    @param text (str): Text to wrap

    @param font (PIL.ImageFont): Font to use for text

    @param max_width (int): Maximum width of the image

    @returns: List of wrapped text, where each element is a line of text
    """
    lines = []

    if font.getbbox(text)[2] <= max_width:
        return [text]
    else:
        words = text.split(" ")
        line_words = []
        lines = []
        for word in words:
            if font.getbbox(" ".join(line_words + [word]))[2] <= max_width:
                line_words.append(word)
            else:
                lines.append(" ".join(line_words))
                line_words = [word]
        lines.append(" ".join(line_words))

    return lines
