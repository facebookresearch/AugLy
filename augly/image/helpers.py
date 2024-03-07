#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable

import numpy as np
from PIL import Image


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
