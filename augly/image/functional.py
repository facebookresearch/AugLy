#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import io
import math
import os
import pickle
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from augly import utils
from augly.image import utils as imutils
from augly.image.utils.bboxes import spatial_bbox_helper
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def apply_lambda(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    aug_function: Callable[..., Image.Image] = lambda x: x,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
    **kwargs,
) -> Image.Image:
    """
    Apply a user-defined lambda on an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param aug_function: the augmentation function to be applied onto the image
        (should expect a PIL image as input and return one)

    @param **kwargs: the input attributes to be passed into the augmentation
        function to be applied

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert callable(aug_function), (
        repr(type(aug_function).__name__) + " object is not callable"
    )
    image = imutils.validate_and_load_image(image)

    func_kwargs = deepcopy(locals())
    if aug_function is not None:
        try:
            func_kwargs["aug_function"] = aug_function.__name__
        except AttributeError:
            func_kwargs["aug_function"] = type(aug_function).__name__

    func_kwargs = imutils.get_func_kwargs(metadata, func_kwargs)
    src_mode = image.mode

    aug_image = aug_function(image, **kwargs)

    imutils.get_metadata(
        metadata=metadata,
        function_name="apply_lambda",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def apply_pil_filter(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    filter_type: Union[Callable, ImageFilter.Filter] = ImageFilter.EDGE_ENHANCE_MORE,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Applies a given PIL filter to the input image using `Image.filter()`

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param filter_type: the PIL ImageFilter to apply to the image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = deepcopy(locals())

    ftr = filter_type() if isinstance(filter_type, Callable) else filter_type
    assert isinstance(
        ftr, ImageFilter.Filter
    ), "Filter type must be a PIL.ImageFilter.Filter class"

    func_kwargs = imutils.get_func_kwargs(
        metadata, func_kwargs, filter_type=getattr(ftr, "name", filter_type)
    )
    src_mode = image.mode

    aug_image = image.filter(ftr)

    imutils.get_metadata(
        metadata=metadata,
        function_name="apply_pil_filter",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def blur(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    radius: float = 2.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Blurs the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param radius: the larger the radius, the blurrier the image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert radius > 0, "Radius cannot be negative"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    aug_image = image.filter(ImageFilter.GaussianBlur(radius))

    imutils.get_metadata(
        metadata=metadata,
        function_name="blur",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def brightness(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Changes the brightness of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: values less than 1.0 darken the image and values greater than 1.0
        brighten the image. Setting factor to 1.0 will not alter the image's brightness

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    aug_image = ImageEnhance.Brightness(image).enhance(factor)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    src_mode = image.mode
    imutils.get_metadata(metadata=metadata, function_name="brightness", **func_kwargs)

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def change_aspect_ratio(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Changes the aspect ratio of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param ratio: aspect ratio, i.e. width/height, of the new image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert ratio > 0, "Ratio cannot be negative"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size
    area = width * height
    new_width = int(math.sqrt(ratio * area))
    new_height = int(area / new_width)
    aug_image = image.resize((new_width, new_height))

    imutils.get_metadata(
        metadata=metadata,
        function_name="change_aspect_ratio",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def clip_image_size(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    min_resolution: Optional[int] = None,
    max_resolution: Optional[int] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Scales the image up or down if necessary to fit in the given min and max resolution

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param min_resolution: the minimum resolution, i.e. width * height, that the
        augmented image should have; if the input image has a lower resolution than this,
        the image will be scaled up as necessary

    @param max_resolution: the maximum resolution, i.e. width * height, that the
        augmented image should have; if the input image has a higher resolution than
        this, the image will be scaled down as necessary

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert min_resolution is None or (
        isinstance(min_resolution, int) and min_resolution >= 0
    ), "min_resolution must be None or a nonnegative int"
    assert max_resolution is None or (
        isinstance(max_resolution, int) and max_resolution >= 0
    ), "max_resolution must be None or a nonnegative int"
    assert not (
        min_resolution is not None
        and max_resolution is not None
        and min_resolution > max_resolution
    ), "min_resolution cannot be greater than max_resolution"

    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode
    aug_image = image

    if min_resolution is not None and image.width * image.height < min_resolution:
        resize_factor = math.sqrt(min_resolution / (image.width * image.height))
        aug_image = scale(aug_image, factor=resize_factor)

    elif max_resolution is not None and image.width * image.height > max_resolution:
        resize_factor = math.sqrt(max_resolution / (image.width * image.height))
        aug_image = scale(aug_image, factor=resize_factor)

    imutils.get_metadata(
        metadata=metadata,
        function_name="clip_image_size",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def color_jitter(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    saturation_factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Color jitters the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param brightness_factor: a brightness factor below 1.0 darkens the image, a factor
        of 1.0 does not alter the image, and a factor greater than 1.0 brightens the image

    @param contrast_factor: a contrast factor below 1.0 removes contrast, a factor of
        1.0 gives the original image, and a factor greater than 1.0 adds contrast

    @param saturation_factor: a saturation factor of below 1.0 lowers the saturation,
        a factor of 1.0 gives the original image, and a factor greater than 1.0
        adds saturation

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    aug_image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    aug_image = ImageEnhance.Contrast(aug_image).enhance(contrast_factor)
    aug_image = ImageEnhance.Color(aug_image).enhance(saturation_factor)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    src_mode = image.mode

    imutils.get_metadata(metadata=metadata, function_name="color_jitter", **func_kwargs)

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def contrast(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Alters the contrast of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: zero gives a grayscale image, values below 1.0 decreases contrast,
        a factor of 1.0 gives the original image, and a factor greater than 1.0
        increases contrast

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: Image.Image - Augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    enhancer = ImageEnhance.Contrast(image)
    aug_image = enhancer.enhance(factor)

    imutils.get_metadata(
        metadata=metadata,
        function_name="contrast",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def convert_color(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    mode: Optional[str] = None,
    matrix: Union[
        None,
        Tuple[float, float, float, float],
        Tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
    ] = None,
    dither: Optional[int] = None,
    palette: int = 0,
    colors: int = 256,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Converts the image in terms of color modes

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param mode: defines the type and depth of a pixel in the image. If mode is omitted,
        a mode is chosen so that all information in the image and the palette can be
        represented without a palette. For list of available modes, check:
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

    @param matrix: an optional conversion matrix. If given, this should be 4- or
        12-tuple containing floating point values

    @param dither: dithering method, used when converting from mode “RGB” to “P” or from
        “RGB” or “L” to “1”. Available methods are NONE or FLOYDSTEINBERG (default).

    @param palette: palette to use when converting from mode “RGB” to “P”. Available
        palettes are WEB or ADAPTIVE

    @param colors: number of colors to use for the ADAPTIVE palette. Defaults to 256.

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: Image.Image - Augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    # pyre-fixme[6]: Expected `Union[typing_extensions.Literal[0],
    #  typing_extensions.Literal[1]]` for 4th param but got `int`.
    aug_image = image.convert(mode, matrix, dither, palette, colors)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    imutils.get_metadata(
        metadata=metadata,
        function_name="convert_color",
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path)


def crop(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    x1: float = 0.25,
    y1: float = 0.25,
    x2: float = 0.75,
    y2: float = 0.75,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Crops the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param x1: position of the left edge of cropped image relative to the width of
        the original image; must be a float value between 0 and 1

    @param y1: position of the top edge of cropped image relative to the height of
        the original image; must be a float value between 0 and 1

    @param x2: position of the right edge of cropped image relative to the width of
        the original image; must be a float value between 0 and 1

    @param y2: position of the bottom edge of cropped image relative to the height of
        the original image; must be a float value between 0 and 1

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0 <= x1 <= 1.0, "x1 must be a value in the range [0, 1]"
    assert 0 <= y1 <= 1.0, "y1 must be a value in the range [0, 1]"
    assert x1 < x2 <= 1.0, "x2 must be a value in the range [x1, 1]"
    assert y1 < y2 <= 1.0, "y2 must be a value in the range [y1, 1]"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size

    left, right = int(width * x1), int(width * x2)
    top, bottom = int(height * y1), int(height * y2)

    aug_image = image.crop((left, top, right, bottom))

    imutils.get_metadata(
        metadata=metadata,
        function_name="crop",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def encoding_quality(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    quality: int = 50,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Changes the JPEG encoding quality level

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param quality: JPEG encoding quality. 0 is lowest quality, 100 is highest

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0 <= quality <= 100, "'quality' must be a value in the range [0, 100]"

    image = imutils.validate_and_load_image(image).convert("RGB")

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    aug_image = Image.open(buffer)

    imutils.get_metadata(
        metadata=metadata,
        function_name="encoding_quality",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def grayscale(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    mode: str = "luminosity",
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Changes an image to be grayscale

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param mode: the type of greyscale conversion to perform; two options
        are supported ("luminosity" and "average")

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert mode in [
        "luminosity",
        "average",
    ], "Greyscale mode not supported -- choose either 'luminosity' or 'average'"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    # If grayscale image is passed in, return it
    if image.mode == "L":
        aug_image = image
    else:
        if mode == "luminosity":
            aug_image = image.convert(mode="L")
        elif mode == "average":
            np_image = np.asarray(image).astype(np.float32)
            np_image = np.average(np_image, axis=2)
            aug_image = Image.fromarray(np.uint8(np_image))
        aug_image = aug_image.convert(mode="RGB")

    imutils.get_metadata(
        metadata=metadata,
        function_name="grayscale",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def hflip(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Horizontally flips an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    aug_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    src_mode = image.mode
    imutils.get_metadata(metadata=metadata, function_name="hflip", **func_kwargs)

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def masked_composite(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    mask: Optional[Union[str, Image.Image]] = None,
    transform_function: Optional[Callable] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Applies given augmentation function to the masked area of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param mask: the path to an image or a variable of type PIL.Image.Image for
        masking. This image can have mode “1”, “L”, or “RGBA”, and must have the
        same size as the other two images. If the mask is not provided the function
        returns the augmented image

    @param transform_function: the augmentation function to be applied. If
        transform_function is not provided, the function returns the input image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = deepcopy(locals())
    if transform_function is not None:
        try:
            func_kwargs["transform_function"] = transform_function.__name__
        except AttributeError:
            func_kwargs["transform_function"] = type(transform_function).__name__
    func_kwargs = imutils.get_func_kwargs(metadata, func_kwargs)
    src_mode = image.mode

    if transform_function is None:
        masked_image = imutils.ret_and_save_image(image, output_path)
    else:
        aug_image = transform_function(image)
        if mask is None:
            masked_image = imutils.ret_and_save_image(aug_image, output_path, src_mode)
        else:
            mask = imutils.validate_and_load_image(mask)
            assert image.size == mask.size, "Mask size must be equal to image size"
            masked_image = Image.composite(aug_image, image, mask)

    imutils.get_metadata(
        metadata=metadata,
        function_name="masked_composite",
        aug_image=masked_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(masked_image, output_path, src_mode)


def meme_format(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    text: str = "LOL",
    font_file: str = utils.MEME_DEFAULT_FONT,
    opacity: float = 1.0,
    text_color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    caption_height: int = 250,
    meme_bg_color: Tuple[int, int, int] = utils.WHITE_RGB_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Creates a new image that looks like a meme, given text and an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param text: the text to be overlaid/used in the meme. note: if using a very
        long string, please add in newline characters such that the text remains
        in a readable font size.

    @param font_file: iopath uri to a .ttf font file

    @param opacity: the lower the opacity, the more transparent the text

    @param text_color: color of the text in RGB values

    @param caption_height: the height of the meme caption

    @param meme_bg_color: background color of the meme caption in RGB values

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert isinstance(text, str), "Expected variable `text` to be a string"
    assert 0.0 <= opacity <= 1.0, "Opacity must be a value in the range [0.0, 1.0]"
    assert caption_height > 10, "Caption height must be greater than 10"

    utils.validate_rgb_color(text_color)
    utils.validate_rgb_color(meme_bg_color)

    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size

    local_font_path = utils.pathmgr.get_local_path(font_file)
    font_size = caption_height - 10

    meme = Image.new("RGB", (width, height + caption_height), meme_bg_color)
    meme.paste(image, (0, caption_height))
    draw = ImageDraw.Draw(meme)

    x_pos, y_pos = 5, 5
    ascender_adjustment = 40
    while True:
        font = ImageFont.truetype(local_font_path, font_size)
        text_bbox = draw.multiline_textbbox(
            (x_pos, y_pos),
            text,
            # pyre-fixme[6]: Expected `Optional[ImageFont._Font]` for 3rd param but got
            #  `FreeTypeFont`.
            font=font,
            anchor="la",
            align="center",
        )

        text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )

        x_pos = round((width - text_width) / 2)
        y_pos = round((caption_height - text_height) / 2) - ascender_adjustment

        if text_width <= (width - 10) and text_height <= (caption_height - 10):
            break

        font_size -= 5

    draw.multiline_text(
        (x_pos, y_pos),
        text,
        # pyre-fixme[6]: Expected `Optional[ImageFont._Font]` for 3rd param but got
        #  `FreeTypeFont`.
        font=font,
        anchor="la",
        fill=(text_color[0], text_color[1], text_color[2], round(opacity * 255)),
        align="center",
    )

    imutils.get_metadata(
        metadata=metadata,
        function_name="meme_format",
        aug_image=meme,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(meme, output_path, src_mode)


def opacity(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    level: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Alter the opacity of an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param level: the level the opacity should be set to, where 0 means
        completely transparent and 1 means no transparency at all

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0 <= level <= 1, "level must be a value in the range [0, 1]"

    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode
    image = image.convert(mode="RGBA")

    mask = image.convert("RGBA").getchannel("A")
    mask = Image.fromarray((np.array(mask) * level).astype(np.uint8))
    background = Image.new("RGBA", image.size, (255, 255, 255, 0))
    aug_image = Image.composite(image, background, mask)

    imutils.get_metadata(
        metadata=metadata,
        function_name="opacity",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def overlay_emoji(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    emoji_path: str = utils.EMOJI_PATH,
    opacity: float = 1.0,
    emoji_size: float = 0.15,
    x_pos: float = 0.4,
    y_pos: float = 0.8,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlay an emoji onto the original image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param emoji_path: iopath uri to the emoji image

    @param opacity: the lower the opacity, the more transparent the overlaid emoji

    @param emoji_size: size of the emoji is emoji_size * height of the original image

    @param x_pos: position of emoji relative to the image width

    @param y_pos: position of emoji relative to the image height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    local_emoji_path = utils.pathmgr.get_local_path(emoji_path)

    aug_image = overlay_image(
        image,
        overlay=local_emoji_path,
        output_path=output_path,
        opacity=opacity,
        overlay_size=emoji_size,
        x_pos=x_pos,
        y_pos=y_pos,
    )

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_emoji",
        aug_image=aug_image,
        **func_kwargs,
    )

    return aug_image


def overlay_image(
    image: Union[str, Image.Image],
    overlay: Union[str, Image.Image],
    output_path: Optional[str] = None,
    opacity: float = 1.0,
    overlay_size: float = 1.0,
    x_pos: float = 0.4,
    y_pos: float = 0.4,
    max_visible_opacity: float = 0.75,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlays an image onto another image at position (width * x_pos, height * y_pos)

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param overlay: the path to an image or a variable of type PIL.Image.Image
        that will be overlaid

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param opacity: the lower the opacity, the more transparent the overlaid image

    @param overlay_size: size of the overlaid image is overlay_size * height
        of the original image

    @param x_pos: position of overlaid image relative to the image width

    @param max_visible_opacity: if bboxes are passed in, this param will be used as the
        maximum opacity value through which the src image will still be considered
        visible; see the function `overlay_image_bboxes_helper` in `utils/bboxes.py` for
        more details about how this is used. If bboxes are not passed in this is not used

    @param y_pos: position of overlaid image relative to the image height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0.0 <= opacity <= 1.0, "Opacity must be a value in the range [0, 1]"
    assert 0.0 <= overlay_size <= 1.0, "Image size must be a value in the range [0, 1]"
    assert 0.0 <= x_pos <= 1.0, "x_pos must be a value in the range [0, 1]"
    assert 0.0 <= y_pos <= 1.0, "y_pos must be a value in the range [0, 1]"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    overlay = imutils.validate_and_load_image(overlay)

    im_width, im_height = image.size
    overlay_width, overlay_height = overlay.size
    new_height = max(1, int(im_height * overlay_size))
    new_width = int(overlay_width * new_height / overlay_height)
    overlay = overlay.resize((new_width, new_height))

    try:
        mask = overlay.convert("RGBA").getchannel("A")
        mask = Image.fromarray((np.array(mask) * opacity).astype(np.uint8))
    except ValueError:
        mask = Image.new(mode="L", size=overlay.size, color=int(opacity * 255))

    x = int(im_width * x_pos)
    y = int(im_height * y_pos)

    aug_image = image.convert(mode="RGBA")
    aug_image.paste(im=overlay, box=(x, y), mask=mask)

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_image",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def overlay_onto_background_image(
    image: Union[str, Image.Image],
    background_image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    opacity: float = 1.0,
    overlay_size: float = 1.0,
    x_pos: float = 0.4,
    y_pos: float = 0.4,
    scale_bg: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlays the image onto a given background image at position
    (width * x_pos, height * y_pos)

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param background_image: the path to an image or a variable of type PIL.Image.Image
        onto which the source image will be overlaid

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param opacity: the lower the opacity, the more transparent the overlaid image

    @param overlay_size: size of the overlaid image is overlay_size * height
        of the background image

    @param x_pos: position of overlaid image relative to the background image width with
        respect to the x-axis

    @param y_pos: position of overlaid image relative to the background image height with
        respect to the y-axis

    @param scale_bg: if True, the background image will be scaled up or down so that
        overlay_size is respected; if False, the source image will be scaled instead

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0.0 <= overlay_size <= 1.0, "Image size must be a value in the range [0, 1]"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    if scale_bg:
        background_image = resize(
            background_image,
            width=math.floor(image.width / overlay_size),
            height=math.floor(image.height / overlay_size),
        )

    aug_image = overlay_image(
        background_image,
        overlay=image,
        output_path=output_path,
        opacity=opacity,
        overlay_size=overlay_size,
        x_pos=x_pos,
        y_pos=y_pos,
    )

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_onto_background_image",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def overlay_onto_screenshot(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    template_filepath: str = utils.TEMPLATE_PATH,
    template_bboxes_filepath: str = utils.BBOXES_PATH,
    max_image_size_pixels: Optional[int] = None,
    crop_src_to_fit: bool = False,
    resize_src_to_match_template: bool = True,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlay the image onto a screenshot template so it looks like it was
    screenshotted on Instagram

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param template_filepath: iopath uri to the screenshot template

    @param template_bboxes_filepath: iopath uri to the file containing the
        bounding box for each template

    @param max_image_size_pixels: if provided, the template image and/or src image
        will be scaled down to avoid an output image with an area greater than this
        size (in pixels)

    @param crop_src_to_fit: if True, the src image will be cropped if necessary to fit
        into the template image if the aspect ratios are different. If False, the src
        image will instead be resized if needed

    @param resize_src_to_match_template: if True, the src image will be resized if it is
        too big or small in both dimensions to better match the template image. If False,
        the template image will be resized to match the src image instead. It can be
        useful to set this to True if the src image is very large so that the augmented
        image isn't huge, but instead is the same size as the template image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    template, bbox = imutils.get_template_and_bbox(
        template_filepath, template_bboxes_filepath
    )

    if resize_src_to_match_template:
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        image = scale(image, factor=min(bbox_w / image.width, bbox_h / image.height))
    else:
        template, bbox = imutils.scale_template_image(
            image.size[0],
            image.size[1],
            template,
            bbox,
            max_image_size_pixels,
            crop_src_to_fit,
        )
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    cropped_src = imutils.resize_and_pad_to_given_size(
        image, bbox_w, bbox_h, crop=crop_src_to_fit
    )
    template.paste(cropped_src, box=bbox)

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_onto_screenshot",
        aug_image=template,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(template, output_path, src_mode)


def overlay_stripes(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    line_width: float = 0.5,
    line_color: Tuple[int, int, int] = utils.WHITE_RGB_COLOR,
    line_angle: float = 0,
    line_density: float = 0.5,
    line_type: Optional[str] = "solid",
    line_opacity: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlay stripe pattern onto the image (by default, white horizontal
    stripes are overlaid)

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param line_width: the width of individual stripes as a float value ranging
        from 0 to 1. Defaults to 0.5

    @param line_color: color of the overlaid stripes in RGB values

    @param line_angle: the angle of the stripes in degrees, ranging from
        -360° to 360°. Defaults to 0° or horizontal stripes

    @param line_density: controls the distance between stripes represented as a
        float value ranging from 0 to 1, with 1 indicating more densely spaced
        stripes. Defaults to 0.5

    @param line_type: the type of stripes. Current options include: dotted,
        dashed, and solid. Defaults to solid

    @param line_opacity: the opacity of the stripes, ranging from 0 to 1 with
        1 being opaque. Defaults to 1.0

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert (
        0.0 <= line_width <= 1.0
    ), "Line width must be a value in the range [0.0, 1.0]"
    assert (
        -360.0 <= line_angle <= 360.0
    ), "Line angle must be a degree in the range [360.0, 360.0]"
    assert (
        0.0 <= line_density <= 1.0
    ), "Line density must be a value in the range [0.0, 1.0]"
    assert (
        0.0 <= line_opacity <= 1.0
    ), "Line opacity must be a value in the range [0.0, 1.0]"
    assert line_type in utils.SUPPORTED_LINE_TYPES, "Stripe type not supported"
    utils.validate_rgb_color(line_color)

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size

    binary_mask = imutils.compute_stripe_mask(
        src_w=width,
        src_h=height,
        line_width=line_width,
        line_angle=line_angle,
        line_density=line_density,
    )

    if line_type == "dotted":
        # To create dotted effect, multiply mask by stripes in perpendicular direction
        perpendicular_mask = imutils.compute_stripe_mask(
            src_w=width,
            src_h=height,
            line_width=line_width,
            line_angle=line_angle + 90,
            line_density=line_density,
        )
        binary_mask *= perpendicular_mask
    elif line_type == "dashed":
        # To create dashed effect, multiply mask by stripes with a larger line
        # width in perpendicular direction
        perpendicular_mask = imutils.compute_stripe_mask(
            src_w=width,
            src_h=height,
            line_width=0.7,
            line_angle=line_angle + 90,
            line_density=line_density,
        )
        binary_mask *= perpendicular_mask

    mask = Image.fromarray(np.uint8(binary_mask * line_opacity * 255))

    foreground = Image.new("RGB", image.size, line_color)
    aug_image = image.copy()  # to avoid modifying the input image
    aug_image.paste(foreground, (0, 0), mask=mask)

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_stripes",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def overlay_text(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    text: List[Union[int, List[int]]] = utils.DEFAULT_TEXT_INDICES,
    font_file: str = utils.FONT_PATH,
    font_size: float = 0.15,
    opacity: float = 1.0,
    color: Tuple[int, int, int] = utils.RED_RGB_COLOR,
    x_pos: float = 0.0,
    y_pos: float = 0.5,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Overlay text onto the image (by default, text is randomly overlaid)

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param text: indices (into the file) of the characters to be overlaid. Each line of
        text is represented as a list of int indices; if a list of lists is supplied,
        multiple lines of text will be overlaid

    @param font_file: iopath uri to the .ttf font file

    @param font_size: size of the overlaid characters, calculated as
        font_size * min(height, width) of the original image

    @param opacity: the lower the opacity, the more transparent the overlaid text

    @param color: color of the overlaid text in RGB values

    @param x_pos: position of the overlaid text relative to the image width

    @param y_pos: position of the overlaid text relative to the image height

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert 0.0 <= opacity <= 1.0, "Opacity must be a value in the range [0.0, 1.0]"
    assert 0.0 <= font_size <= 1.0, "Font size must be a value in the range [0.0, 1.0]"
    assert 0.0 <= x_pos <= 1.0, "x_pos must be a value in the range [0.0, 1.0]"
    assert 0.0 <= y_pos <= 1.0, "y_pos must be a value in the range [0.0, 1.0]"
    utils.validate_rgb_color(color)

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    text_lists = text if all(isinstance(t, list) for t in text) else [text]
    assert all(isinstance(t, list) for t in text_lists) and all(
        all(isinstance(t, int) for t in text_l)  # pyre-ignore text_l is a List[int]
        for text_l in text_lists
    ), "Text must be a list of ints or a list of list of ints for multiple lines"

    image = image.convert("RGBA")
    width, height = image.size

    local_font_path = utils.pathmgr.get_local_path(font_file)
    font_size = int(min(width, height) * font_size)
    font = ImageFont.truetype(local_font_path, font_size)

    pkl_file = os.path.splitext(font_file)[0] + ".pkl"
    local_pkl_path = utils.pathmgr.get_local_path(pkl_file)

    with open(local_pkl_path, "rb") as f:
        chars = pickle.load(f)

    try:
        text_strs = [
            # pyre-fixme[16]: Item `int` of `Union[List[int], List[Union[List[int],
            #  int]], int]` has no attribute `__iter__`.
            "".join([chr(chars[c % len(chars)]) for c in t])
            for t in text_lists
        ]
    except Exception:
        raise IndexError("Invalid text indices specified")

    draw = ImageDraw.Draw(image)
    for i, text_str in enumerate(text_strs):
        draw.text(
            xy=(x_pos * width, y_pos * height + i * (font_size + 5)),
            text=text_str,
            fill=(color[0], color[1], color[2], round(opacity * 255)),
            # pyre-fixme[6]: Expected `Optional[ImageFont._Font]` for 4th param but got
            #  `FreeTypeFont`.
            font=font,
        )

    imutils.get_metadata(
        metadata=metadata,
        function_name="overlay_text",
        aug_image=image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(image, output_path, src_mode)


def pad(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    w_factor: float = 0.25,
    h_factor: float = 0.25,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Pads the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param w_factor: width * w_factor pixels are padded to both left and right
        of the image

    @param h_factor: height * h_factor pixels are padded to the top and the
        bottom of the image

    @param color: color of the padded border in RGB values

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert w_factor >= 0, "w_factor cannot be a negative number"
    assert h_factor >= 0, "h_factor cannot be a negative number"
    utils.validate_rgb_color(color)

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size

    left = right = int(w_factor * width)
    top = bottom = int(h_factor * height)

    aug_image = Image.new(
        mode="RGB",
        size=(width + left + right, height + top + bottom),
        color=color,
    )
    aug_image.paste(image, box=(left, top))

    imutils.get_metadata(
        metadata=metadata,
        function_name="pad",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def pad_square(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Pads the shorter edge of the image such that it is now square-shaped

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param color: color of the padded border in RGB values

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    utils.validate_rgb_color(color)
    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    width, height = image.size

    if width < height:
        h_factor = 0
        dw = height - width
        w_factor = dw / (2 * width)
    else:
        w_factor = 0
        dh = width - height
        h_factor = dh / (2 * height)

    aug_image = pad(image, output_path, w_factor, h_factor, color)

    imutils.get_metadata(
        metadata=metadata,
        function_name="pad_square",
        aug_image=aug_image,
        **func_kwargs,
    )

    return aug_image


def perspective_transform(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    sigma: float = 50.0,
    dx: float = 0.0,
    dy: float = 0.0,
    seed: Optional[int] = 42,
    crop_out_black_border: bool = False,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Apply a perspective transform to the image so it looks like it was taken
    as a photo from another device (e.g. taking a picture from your phone of a
    picture on a computer).

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param sigma: the standard deviation of the distribution of destination
        coordinates. the larger the sigma value, the more intense the transform

    @param dx: change in x for the perspective transform; instead of providing
        `sigma` you can provide a scalar value to be precise

    @param dy: change in y for the perspective transform; instead of providing
        `sigma` you can provide a scalar value to be precise

    @param seed: if provided, this will set the random seed to ensure consistency
        between runs

    @param crop_out_black_border: if True, will crop out the black border resulting
        from the perspective transform by cropping to the largest center rectangle
        with no black

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert sigma >= 0, "Expected 'sigma' to be nonnegative"
    assert isinstance(dx, (int, float)), "Expected 'dx' to be a number"
    assert isinstance(dy, (int, float)), "Expected 'dy' to be a number"
    assert seed is None or isinstance(
        seed, int
    ), "Expected 'seed' to be an integer or set to None"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    rng = np.random.RandomState(seed) if seed is not None else np.random
    width, height = image.size

    src_coords = [(0, 0), (width, 0), (width, height), (0, height)]
    dst_coords = [
        (rng.normal(point[0], sigma) + dx, rng.normal(point[1], sigma) + dy)
        for point in src_coords
    ]

    perspective_transform_coeffs = imutils.compute_transform_coeffs(
        src_coords, dst_coords
    )
    aug_image = image.transform(
        (width, height), Image.PERSPECTIVE, perspective_transform_coeffs, Image.BICUBIC
    )

    if crop_out_black_border:
        top_left, top_right, bottom_right, bottom_left = dst_coords
        new_left = max(0, top_left[0], bottom_left[0])
        new_right = min(width, top_right[0], bottom_right[0])
        new_top = max(0, top_left[1], top_right[1])
        new_bottom = min(height, bottom_left[1], bottom_right[1])

        if new_left >= new_right or new_top >= new_bottom:
            raise Exception(
                "Cannot crop out black border of a perspective transform this intense"
            )

        aug_image = aug_image.crop((new_left, new_top, new_right, new_bottom))

    imutils.get_metadata(
        metadata=metadata,
        function_name="perspective_transform",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def pixelization(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Pixelizes an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param ratio: smaller values result in a more pixelated image, 1.0 indicates
        no change, and any value above one doesn't have a noticeable effect

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert ratio > 0, "Expected 'ratio' to be a positive number"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    width, height = image.size
    aug_image = image.resize((int(width * ratio), int(height * ratio)))
    aug_image = aug_image.resize((width, height))

    imutils.get_metadata(
        metadata=metadata,
        function_name="pixelization",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def random_noise(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    mean: float = 0.0,
    var: float = 0.01,
    seed: int = 42,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Adds random noise to the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param mean: mean of the gaussian noise added

    @param var: variance of the gaussian noise added

    @param seed: if provided, this will set the random seed before generating noise

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert type(mean) in [float, int], "Mean must be an integer or a float"
    assert type(var) in [float, int], "Variance must be an integer or a float"
    assert type(seed) == int, "Seed must be an integer"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    if seed is not None:
        np.random.seed(seed=seed)

    np_image = np.asarray(image).astype(np.float32)
    np_image = np_image / 255.0

    if np_image.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0

    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (np_image.shape))
    noisy_image = np_image + gauss
    noisy_image = np.clip(noisy_image, low_clip, 1.0)

    noisy_image *= 255.0

    aug_image = Image.fromarray(np.uint8(noisy_image))

    imutils.get_metadata(
        metadata=metadata,
        function_name="random_noise",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def resize(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resample: Any = Image.BILINEAR,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Resizes an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param width: the desired width the image should be resized to have. If
        None, the original image width will be used

    @param height: the desired height the image should be resized to have. If
        None, the original image height will be used

    @param resample: A resampling filter. This can be one of PIL.Image.NEAREST,
        PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC, or
        PIL.Image.LANCZOS

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert width is None or type(width) == int, "Width must be an integer"
    assert height is None or type(height) == int, "Height must be an integer"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    im_w, im_h = image.size
    aug_image = image.resize((width or im_w, height or im_h), resample)

    imutils.get_metadata(
        metadata=metadata,
        function_name="resize",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def rotate(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    degrees: float = 15.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Rotates the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param degrees: the amount of degrees that the original image will be rotated
        counter clockwise

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert type(degrees) in [float, int], "Degrees must be an integer or a float"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    rotated_image = image.rotate(degrees, expand=True)

    center_x, center_y = rotated_image.width / 2, rotated_image.height / 2
    wr, hr = imutils.rotated_rect_with_max_area(image.width, image.height, degrees)
    aug_image = rotated_image.crop(
        (
            int(center_x - wr / 2),
            int(center_y - hr / 2),
            int(center_x + wr / 2),
            int(center_y + hr / 2),
        )
    )

    imutils.get_metadata(
        metadata=metadata,
        function_name="rotate",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def saturation(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Alters the saturation of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: a saturation factor of below 1.0 lowers the saturation, a
        factor of 1.0 gives the original image, and a factor greater than 1.0
        adds saturation

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    aug_image = ImageEnhance.Color(image).enhance(factor)

    imutils.get_metadata(
        metadata=metadata,
        function_name="saturation",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def scale(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 0.5,
    interpolation: Optional[int] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Alters the resolution of an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: the ratio by which the image should be downscaled or upscaled

    @param interpolation: interpolation method. This can be one of PIL.Image.NEAREST,
        PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC or
        PIL.Image.LANCZOS

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    assert factor > 0, "Expected 'factor' to be a positive number"
    assert interpolation in [
        Image.NEAREST,
        Image.BOX,
        Image.BILINEAR,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        None,
    ], "Invalid interpolation specified"

    image = imutils.validate_and_load_image(image)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    if interpolation is None:
        interpolation = Image.LANCZOS if factor < 1 else Image.BILINEAR

    width, height = image.size

    scaled_width = int(width * factor)
    scaled_height = int(height * factor)

    # pyre-fixme[6]: Expected `Union[typing_extensions.Literal[0],
    #  typing_extensions.Literal[1], typing_extensions.Literal[2],
    #  typing_extensions.Literal[3], typing_extensions.Literal[4],
    #  typing_extensions.Literal[5], None]` for 2nd param but got `int`.
    aug_image = image.resize((scaled_width, scaled_height), resample=interpolation)

    imutils.get_metadata(
        metadata=metadata,
        function_name="scale",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def sharpen(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 1.0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Changes the sharpness of the image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: a factor of below 1.0 blurs the image, a factor of 1.0 gives
        the original image, and a factor greater than 1.0 sharpens the image

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    aug_image = ImageEnhance.Sharpness(image).enhance(factor)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode
    imutils.get_metadata(metadata=metadata, function_name="sharpen", **func_kwargs)

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def shuffle_pixels(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    factor: float = 1.0,
    seed: int = 10,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Shuffles the pixels of an image with respect to the shuffling factor. The
    factor denotes percentage of pixels to be shuffled and randomly selected
    Note: The actual number of pixels will be less than the percentage given
    due to the probability of pixels staying in place in the course of shuffling

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param factor: a control parameter between 0.0 and 1.0. While a factor of
        0.0 returns the original image, a factor of 1.0 performs full shuffling

    @param seed: seed for numpy random generator to select random pixels for shuffling

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    np.random.seed(seed)

    image = imutils.validate_and_load_image(image)

    assert 0.0 <= factor <= 1.0, "'factor' must be a value in range [0, 1]"

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    if factor == 0.0:
        aug_image = image
    else:
        aug_image = np.asarray(image, dtype=int)
        height, width = aug_image.shape[:2]
        number_of_channels = aug_image.size // (height * width)

        number_of_pixels = height * width
        aug_image = np.reshape(aug_image, (number_of_pixels, number_of_channels))

        mask = np.random.choice(
            number_of_pixels, size=int(factor * number_of_pixels), replace=False
        )
        pixels_to_be_shuffled = aug_image[mask]

        np.random.shuffle(pixels_to_be_shuffled)
        aug_image[mask] = pixels_to_be_shuffled

        aug_image = np.reshape(aug_image, (height, width, number_of_channels))
        aug_image = np.squeeze(aug_image)

        aug_image = Image.fromarray(aug_image.astype("uint8"))

    imutils.get_metadata(
        metadata=metadata,
        function_name="shuffle_pixels",
        aug_image=aug_image,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def skew(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    skew_factor: float = 0.5,
    axis: int = 0,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Skews an image with respect to its x or y-axis

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param skew_factor: the level of skew to apply to the image; a larger absolute value will
        result in a more intense skew. Recommended range is between [-2, 2]

    @param axis: the axis along which the image will be skewed; can be set to 0 (x-axis)
        or 1 (y-axis)

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode

    w, h = image.size

    if axis == 0:
        data = (1, skew_factor, -skew_factor * h / 2, 0, 1, 0)
    elif axis == 1:
        data = (1, 0, 0, skew_factor, 1, -skew_factor * w / 2)
    else:
        raise AssertionError(
            f"Invalid 'axis' value: Got '{axis}', expected 0 for 'x-axis' or 1 for 'y-axis'"
        )

    aug_image = image.transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR)
    imutils.get_metadata(
        metadata=metadata,
        function_name="skew",
        aug_image=aug_image,
        bboxes_helper_func=spatial_bbox_helper,
        aug_function=skew,
        **func_kwargs,
    )

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)


def vflip(
    image: Union[str, Image.Image],
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,
) -> Image.Image:
    """
    Vertically flips an image

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: the augmented PIL Image
    """
    image = imutils.validate_and_load_image(image)
    aug_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    func_kwargs = imutils.get_func_kwargs(metadata, locals())
    src_mode = image.mode
    imutils.get_metadata(metadata=metadata, function_name="vflip", **func_kwargs)

    return imutils.ret_and_save_image(aug_image, output_path, src_mode)
