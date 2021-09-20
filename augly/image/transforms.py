#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import augly.image.functional as F
import augly.utils as utils
from PIL import Image, ImageFilter


"""
Base Classes for Transforms
"""


class BaseTransform(object):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        force: bool = False,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        @param image: PIL Image to be augmented

        @param force: if set to True, the transform will be applied. Otherwise,
            application is determined by the probability set

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        assert isinstance(image, Image.Image), "Image passed in must be a PIL Image"
        assert type(force) == bool, "Expected type bool for variable `force`"

        if not force and random.random() > self.p:
            return image

        return self.apply_transform(image, metadata, bboxes, bbox_format)

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()


class BaseRandomRangeTransform(BaseTransform):
    def __init__(self, min_val: float, max_val: float, p: float = 1.0):
        """
        @param min_val: the lower value of the range

        @param max_val: the upper value of the range

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.min_val = min_val
        self.max_val = max_val
        self.chosen_value = None

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        self.chosen_value = (
            random.random() * (self.max_val - self.min_val)
        ) + self.min_val
        return self.apply_random_transform(
            image,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        This function is to be implemented in the child classes. It has
        access to `self.chosen_value` which is the randomly chosen value
        from the range specified to pass into the augmentation function
        """
        raise NotImplementedError()


"""
Non-Random Transforms

These classes below are essentially class-based versions of the augmentation
functions previously defined. These classes were developed such that they can
be used with Composition operators (such as `torchvision`'s) and to support
use cases where a specific transform with specific attributes needs to be
applied multiple times.

Example:
 >>> image = Image.open("....")
 >>> blur_tsfm = Blur(radius=5.0, p=0.5)
 >>> blurred_image = blur_tsfm(image)
"""


class ApplyLambda(BaseTransform):
    def __init__(
        self,
        aug_function: Callable[..., Image.Image] = lambda x: x,
        p: float = 1.0,
        **kwargs,
    ):
        """
        @param aug_function: the augmentation function to be applied onto the image
            (should expect a PIL image as input and return one)

        @param p: the probability of the transform being applied; default value is 1.0

        @param **kwargs: the input attributes to be passed into the augmentation
            function to be applied
        """
        super().__init__(p)
        self.aug_function = aug_function
        self.kwargs = kwargs

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Apply a user-defined lambda on an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.apply_lambda(
            image,
            aug_function=self.aug_function,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
            **self.kwargs,
        )


class ApplyPILFilter(BaseTransform):
    def __init__(
        self,
        filter_type: Union[Callable, ImageFilter.Filter] = ImageFilter.EDGE_ENHANCE_MORE,
        p: float = 1.0,
    ):
        """
        @param filter_type: the PIL ImageFilter to apply to the image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.filter_type = filter_type

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Applies a given PIL filter to the input image using `Image.filter()`

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.apply_pil_filter(
            image,
            filter_type=self.filter_type,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Blur(BaseTransform):
    def __init__(self, radius: float = 2.0, p: float = 1.0):
        """
        @param radius: the larger the radius, the blurrier the image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.radius = radius

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Blurs the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.blur(
            image,
            radius=self.radius,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Brightness(BaseTransform):
    def __init__(self, factor: float = 1.0, p: float = 1.0):
        """
        @param factor: values less than 1.0 darken the image and values greater than
            1.0 brighten the image. Setting factor to 1.0 will not alter the image's
            brightness

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the brightness of the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.brightness(
            image,
            factor=self.factor,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class ChangeAspectRatio(BaseTransform):
    def __init__(self, ratio: float = 1.0, p: float = 1.0):
        """
        @param ratio: aspect ratio, i.e. width/height, of the new image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.ratio = ratio

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the aspect ratio of the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.change_aspect_ratio(
            image,
            ratio=self.ratio,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class ClipImageSize(BaseTransform):
    def __init__(
        self,
        min_resolution: Optional[int] = None,
        max_resolution: Optional[int] = None,
        p: float = 1.0,
    ):
        """
        @param min_resolution: the minimum resolution, i.e. width * height, that the
            augmented image should have; if the input image has a lower resolution than this,
            the image will be scaled up as necessary

        @param max_resolution: the maximum resolution, i.e. width * height, that the
            augmented image should have; if the input image has a higher resolution than
            this, the image will be scaled down as necessary

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Scales the image up or down if necessary to fit in the given min and max
        resolution

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.clip_image_size(
            image,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class ColorJitter(BaseTransform):
    def __init__(
        self,
        brightness_factor: float = 1.0,
        contrast_factor: float = 1.0,
        saturation_factor: float = 1.0,
        p: float = 1.0,
    ):
        """
        @param brightness_factor: a brightness factor below 1.0 darkens the image,
            a factor of 1.0 does not alter the image, and a factor greater than 1.0
            brightens the image

        @param contrast_factor: a contrast factor below 1.0 removes contrast, a factor
            of 1.0 gives the original image, and a factor greater than 1.0 adds contrast

        @param saturation_factor: a saturation factor of below 1.0 lowers the saturation,
            a factor of 1.0 gives the original image, and a factor greater than 1.0 adds
            saturation

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Color jitters the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.color_jitter(
            image,
            brightness_factor=self.brightness_factor,
            contrast_factor=self.contrast_factor,
            saturation_factor=self.saturation_factor,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Contrast(BaseTransform):
    def __init__(self, factor: float = 1.0, p: float = 1.0):
        """
        @param factor: zero gives a grayscale image, values below 1.0 decrease contrast,
            a factor of 1.0 gives the original image, and a factor greater than 1.0
            increases contrast

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the contrast of the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.contrast(
            image,
            factor=self.factor,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format)


class ConvertColor(BaseTransform):
    def __init__(
        self,
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
        p: float = 1.0,
    ):
        """
        @param mode: defines the type and depth of a pixel in the image. If mode is
            omitted, a mode is chosen so that all information in the image and the
            palette can be represented without a palette. For list of available modes,
            check: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

        @param matrix: an optional conversion matrix. If given, this should be 4- or
            12-tuple containing floating point values.

        @param dither: dithering method, used when converting from mode “RGB” to “P” or
            from “RGB” or “L” to “1”. Available methods are NONE or FLOYDSTEINBERG (default)

        @param palette: palette to use when converting from mode “RGB” to “P”. Available
            palettes are WEB or ADAPTIVE

        @param colors: number of colors to use for the ADAPTIVE palette. Defaults to 256

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.mode = mode
        self.matrix = matrix
        self.dither = dither
        self.palette = palette
        self.colors = colors

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Converts the image in terms of color modes

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.convert_color(
            image,
            mode=self.mode,
            matrix=self.matrix,
            dither=self.dither,
            palette=self.palette,
            colors=self.colors,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Crop(BaseTransform):
    def __init__(
        self,
        x1: float = 0.25,
        y1: float = 0.25,
        x2: float = 0.75,
        y2: float = 0.75,
        p: float = 1.0,
    ):
        """
        @param x1: position of the left edge of cropped image relative to the width
            of the original image; must be a float value between 0 and 1

        @param y1: position of the top edge of cropped image relative to the height
            of the original image; must be a float value between 0 and 1

        @param x2: position of the right edge of cropped image relative to the width
            of the original image; must be a float value between 0 and 1

        @param y2: position of the bottom edge of cropped image relative to the height
            of the original image; must be a float value between 0 and 1

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Crops the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.crop(
            image,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class EncodingQuality(BaseTransform):
    def __init__(self, quality: int = 50, p: float = 1.0):
        """
        @param quality: JPEG encoding quality. 0 is lowest quality, 100 is highest

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.quality = quality

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Changes the JPEG encoding quality level

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.encoding_quality(
            image,
            quality=self.quality,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Grayscale(BaseTransform):
    def __init__(self, mode: str = "luminosity", p: float = 1.0):
        """
        @param mode: the type of greyscale conversion to perform; two options are
            supported ("luminosity" and "average")

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.mode = mode

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters an image to be grayscale

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.grayscale(
            image,
            mode=self.mode,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class HFlip(BaseTransform):
    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Horizontally flips an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.hflip(
            image,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class MaskedComposite(BaseTransform):
    def __init__(
        self, transform_function: BaseTransform, mask: Image.Image, p: float = 1.0
    ):
        """
        @param mask: the path to an image or a variable of type PIL.Image.Image for
            masking. This image can have mode “1”, “L”, or “RGBA”, and must have the
            same size as the other two images. If the mask is not provided the function
            returns the augmented image

        @param transform_function: the augmentation function to be applied. If
            transform_function is not provided, function returns the input image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.mask = mask
        self.transform_function = transform_function

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Applies given augmentation function to the masked area of the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.masked_composite(
            image,
            mask=self.mask,
            transform_function=self.transform_function,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class MemeFormat(BaseTransform):
    def __init__(
        self,
        text: str = "LOL",
        font_file: str = utils.MEME_DEFAULT_FONT,
        opacity: float = 1.0,
        text_color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
        caption_height: int = 250,
        meme_bg_color: Tuple[int, int, int] = utils.WHITE_RGB_COLOR,
        p: float = 1.0,
    ):
        """
        @param text: the text to be overlaid/used in the meme. note: if using a very
            long string, please add in newline characters such that the text remains
            in a readable font size

        @param font_file: iopath uri to the .ttf font file

        @param opacity: the lower the opacity, the more transparent the text

        @param text_color: color of the text in RGB values

        @param caption_height: the height of the meme caption

        @param meme_bg_color: background color of the meme caption in RGB values

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.text = text
        self.text_color, self.opacity = text_color, opacity
        self.caption_height, self.meme_bg_color = caption_height, meme_bg_color
        self.font_file = font_file

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Creates a new image that looks like a meme, given text and an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.meme_format(
            image,
            text=self.text,
            font_file=self.font_file,
            opacity=self.opacity,
            text_color=self.text_color,
            caption_height=self.caption_height,
            meme_bg_color=self.meme_bg_color,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Opacity(BaseTransform):
    def __init__(self, level: float = 1.0, p: float = 1.0):
        """
        @param level: the level the opacity should be set to, where 0 means completely
            transparent and 1 means no transparency at all

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.level = level

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the opacity of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.opacity(
            image,
            level=self.level,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class OverlayEmoji(BaseTransform):
    def __init__(
        self,
        emoji_path: str = utils.EMOJI_PATH,
        opacity: float = 1.0,
        emoji_size: float = 0.15,
        x_pos: float = 0.4,
        y_pos: float = 0.8,
        p: float = 1.0,
    ):
        """
        @param emoji_path: iopath uri to the emoji image

        @param opacity: the lower the opacity, the more transparent the overlaid emoji

        @param emoji_size: size of the emoji is emoji_size * height of the original image

        @param x_pos: position of emoji relative to the image width

        @param y_pos: position of emoji relative to the image height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.emoji_path = emoji_path
        self.emoji_size, self.opacity = emoji_size, opacity
        self.x_pos, self.y_pos = x_pos, y_pos

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlay an emoji onto the original image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_emoji(
            image,
            emoji_path=self.emoji_path,
            opacity=self.opacity,
            emoji_size=self.emoji_size,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class OverlayImage(BaseTransform):
    def __init__(
        self,
        overlay: Union[str, Image.Image],
        opacity: float = 1.0,
        overlay_size: float = 1.0,
        x_pos: float = 0.4,
        y_pos: float = 0.4,
        max_visible_opacity: float = 0.75,
        p: float = 1.0,
    ):
        """
        @param overlay: the path to an image or a variable of type PIL.Image.Image
            that will be overlaid

        @param opacity: the lower the opacity, the more transparent the overlaid image

        @param overlay_size: size of the overlaid image is overlay_size * height
            of the original image

        @param x_pos: position of overlaid image relative to the image width

        @param y_pos: position of overlaid image relative to the image height

        @param max_visible_opacity: if bboxes are passed in, this param will be used as
            the maximum opacity value through which the src image will still be
            considered visible; see the function `overlay_image_bboxes_helper` in
            `utils/bboxes.py` for more details about how this is used. If bboxes are not
            passed in this is not used

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.overlay = overlay
        self.overlay_size, self.opacity = overlay_size, opacity
        self.x_pos, self.y_pos = x_pos, y_pos
        self.max_visible_opacity = max_visible_opacity

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlays an image onto another image at position (width * x_pos, height * y_pos)

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_image(
            image,
            overlay=self.overlay,
            opacity=self.opacity,
            overlay_size=self.overlay_size,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            max_visible_opacity=self.max_visible_opacity,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class OverlayOntoBackgroundImage(BaseTransform):
    def __init__(
        self,
        background_image: Union[str, Image.Image],
        opacity: float = 1.0,
        overlay_size: float = 1.0,
        x_pos: float = 0.4,
        y_pos: float = 0.4,
        scale_bg: bool = False,
        p: float = 1.0,
    ):
        """
        @param background_image: the path to an image or a variable of type
            PIL.Image.Image onto which the source image will be overlaid

        @param opacity: the lower the opacity, the more transparent the overlaid image

        @param overlay_size: size of the overlaid image is overlay_size * height
            of the background image

        @param x_pos: position of overlaid image relative to the background image width
            with respect to the x-axis

        @param y_pos: position of overlaid image relative to the background image height
            with respect to the y-axis

        @param scale_bg: if True, the background image will be scaled up or down so that
            overlay_size is respected; if False, the source image will be scaled instead

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.background_image = background_image
        self.opacity = opacity
        self.overlay_size = overlay_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.scale_bg = scale_bg

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlays the image onto a given background image at position
        (width * x_pos, height * y_pos)

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_onto_background_image(
            image,
            background_image=self.background_image,
            opacity=self.opacity,
            overlay_size=self.overlay_size,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            scale_bg=self.scale_bg,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class OverlayOntoScreenshot(BaseTransform):
    def __init__(
        self,
        template_filepath: str = utils.TEMPLATE_PATH,
        template_bboxes_filepath: str = utils.BBOXES_PATH,
        max_image_size_pixels: Optional[int] = None,
        crop_src_to_fit: bool = False,
        resize_src_to_match_template: bool = True,
        p: float = 1.0,
    ):
        """
        @param template_filepath: iopath uri to the screenshot template

        @param template_bboxes_filepath: iopath uri to the file containing the
            bounding box for each template

        @param max_image_size_pixels: if provided, the template image and/or src image
            will be scaled down to avoid an output image with an area greater than this
            size (in pixels)

        @param crop_src_to_fit: if True, the src image will be cropped if necessary to
            fit into the template image if the aspect ratios are different. If False, the
            src image will instead be resized if needed

        @param resize_src_to_match_template: if True, the src image will be resized if it
            is too big or small in both dimensions to better match the template image. If
            False, the template image will be resized to match the src image instead. It
            can be useful to set this to True if the src image is very large so that the
            augmented image isn't huge, but instead is the same size as the template image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.template_filepath = template_filepath
        self.template_bboxes_filepath = template_bboxes_filepath
        self.max_image_size_pixels = max_image_size_pixels
        self.crop_src_to_fit = crop_src_to_fit
        self.resize_src_to_match_template = resize_src_to_match_template

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlay the image onto a screenshot template so it looks like it was
        screenshotted on Instagram

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_onto_screenshot(
            image,
            template_filepath=self.template_filepath,
            template_bboxes_filepath=self.template_bboxes_filepath,
            max_image_size_pixels=self.max_image_size_pixels,
            crop_src_to_fit=self.crop_src_to_fit,
            resize_src_to_match_template=self.resize_src_to_match_template,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class OverlayStripes(BaseTransform):
    def __init__(
        self,
        line_width: float = 0.5,
        line_color: Tuple[int, int, int] = utils.WHITE_RGB_COLOR,
        line_angle: float = 0,
        line_density: float = 0.5,
        line_type: Optional[str] = "solid",
        line_opacity: float = 1.0,
        p: float = 1.0
    ):
        """
        @param line_width: the width of individual stripes as a float value ranging
            from 0 to 1. Defaults to 0.5

        @param line_color: color of the overlaid lines in RGB values

        @param line_angle: the angle of the stripes in degrees, ranging from
            -360° to 360°. Defaults to 0° or horizontal stripes

        @param line_density: controls the distance between stripes represented
            as a float value ranging from 0 to 1, with 1 indicating more densely
            spaced stripes. Defaults to 0.5

        @param line_type: the type of stripes. Current options include: dotted,
            dashed, and solid. Defaults to solid

        @param line_opacity: the opacity of the stripes, ranging from 0 to 1 with
            1 being opaque. Defaults to 1

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.line_width, self.line_angle = line_width, line_angle
        self.line_color, self.line_opacity = line_color, line_opacity
        self.line_density = line_density
        self.line_type = line_type

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlay stripe pattern onto the image (by default, stripes are horizontal)

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_stripes(
            image,
            line_width=self.line_width,
            line_color=self.line_color,
            line_angle=self.line_angle,
            line_density=self.line_density,
            line_type=self.line_type,
            line_opacity=self.line_opacity,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format
        )


class OverlayText(BaseTransform):
    def __init__(
        self,
        text: List[Union[int, List[int]]] = utils.DEFAULT_TEXT_INDICES,
        font_file: str = utils.FONT_PATH,
        font_size: float = 0.15,
        opacity: float = 1.0,
        color: Tuple[int, int, int] = utils.RED_RGB_COLOR,
        x_pos: float = 0.0,
        y_pos: float = 0.5,
        p: float = 1.0,
    ):
        """
        @param text: indices (into the file) of the characters to be overlaid. Each line
            of text is represented as a list of int indices; if a list of lists is
            supplied, multiple lines of text will be overlaid

        @param font_file: iopath uri to the .ttf font file

        @param font_size: size of the overlaid characters, calculated as
            font_size * min(height, width) of the original image

        @param opacity: the lower the opacity, the more transparent the overlaid text

        @param color: color of the overlaid text in RGB values

        @param x_pos: position of the overlaid text relative to the image width

        @param y_pos: position of the overlaid text relative to the image height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.text, self.color = text, color
        self.font_file = font_file
        self.font_size, self.opacity = font_size, opacity
        self.x_pos, self.y_pos = x_pos, y_pos

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Overlay text onto the image (by default, text is randomly overlaid)

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.overlay_text(
            image,
            text=self.text,
            font_file=self.font_file,
            font_size=self.font_size,
            opacity=self.opacity,
            color=self.color,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Pad(BaseTransform):
    def __init__(
        self,
        w_factor: float = 0.25,
        h_factor: float = 0.25,
        color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
        p: float = 1.0,
    ):
        """
        @param w_factor: width * w_factor pixels are padded to both left and
            right of the image

        @param h_factor: height * h_factor pixels are padded to the top and
            the bottom of the image

        @param color: color of the padded border in RGB values

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.w_factor = w_factor
        self.h_factor = h_factor
        self.color = color

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Pads the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.pad(
            image,
            w_factor=self.w_factor,
            h_factor=self.h_factor,
            color=self.color,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class PadSquare(BaseTransform):
    def __init__(
        self, color: Tuple[int, int, int] = utils.DEFAULT_COLOR, p: float = 1.0
    ):
        """
        @param color: color of the padded border in RGB values

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.color = color

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Pads the shorter edge of the image such that it is now square-shaped

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.pad_square(
            image,
            color=self.color,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class PerspectiveTransform(BaseTransform):
    def __init__(
        self,
        sigma: float = 50.0,
        dx: float = 0.0,
        dy: float = 0.0,
        seed: Optional[int] = 42,
        p: float = 1.0,
    ):
        """
        @param sigma: the standard deviation of the distribution of destination
            coordinates. the larger the sigma value, the more intense the transform

        @param seed: if provided, this will set the random seed to ensure
            consistency between runs

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.sigma = sigma
        self.dx, self.dy = dx, dy
        self.seed = seed

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Apply a perspective transform to the image so it looks like it was taken
        as a photo from another device (e.g. taking a picture from your phone of a
        picture on a computer).

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.perspective_transform(
            image,
            sigma=self.sigma,
            dx=self.dx,
            dy=self.dy,
            seed=self.seed,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Pixelization(BaseTransform):
    def __init__(self, ratio: float = 1.0, p: float = 1.0):
        """
        @param ratio: smaller values result in a more pixelated image, 1.0 indicates
            no change, and any value above one doesn't have a noticeable effect

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.ratio = ratio

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Pixelizes an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.pixelization(
            image,
            ratio=self.ratio,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomNoise(BaseTransform):
    def __init__(
        self, mean: float = 0.0, var: float = 0.01, seed: int = 42, p: float = 1.0
    ):
        """
        @param mean: mean of the gaussian noise added

        @param var: variance of the gaussian noise added

        @param seed: if provided, this will set the random seed before generating noise

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.mean = mean
        self.var = var
        self.seed = seed

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Adds random noise to the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.random_noise(
            image,
            mean=self.mean,
            var=self.var,
            seed=self.seed,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Resize(BaseTransform):
    def __init__(
        self, width: Optional[int] = None, height: Optional[int] = None, p: float = 1.0
    ):
        """
        @param width: the desired width the image should be resized to have. If None,
            the original image width will be used

        @param height: the desired height the image should be resized to have. If None,
            the original image height will be used

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.width, self.height = width, height

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Resizes an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.resize(
            image,
            width=self.width,
            height=self.height,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Rotate(BaseTransform):
    def __init__(self, degrees: float = 15.0, p: float = 1.0):
        """
        @param degrees: the amount of degrees that the original image will be rotated
            counter clockwise

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.degrees = degrees

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Rotates the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.rotate(
            image,
            degrees=self.degrees,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Saturation(BaseTransform):
    def __init__(self, factor: float = 1.0, p: float = 1.0):
        """
        @param factor: a saturation factor of below 1.0 lowers the saturation, a
            factor of 1.0 gives the original image, and a factor greater than 1.0
            adds saturation

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the saturation of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.saturation(
            image,
            factor=self.factor,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Scale(BaseTransform):
    def __init__(
        self,
        factor: float = 0.5,
        interpolation: Optional[int] = None,
        p: float = 1.0,
    ):
        """
        @param scale_factor: the ratio by which the image should be down-scaled
            or upscaled

        @param interpolation: interpolation method. This can be one of PIL.Image.NEAREST,
            PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC or
            PIL.Image.LANCZOS

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor
        self.interpolation = interpolation

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the resolution of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.scale(
            image,
            factor=self.factor,
            interpolation=self.interpolation,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class Sharpen(BaseTransform):
    def __init__(self, factor: float = 1.0, p: float = 1.0):
        """
        @param factor: a factor of below 1.0 blurs the image, a factor of 1.0 gives
            the original image, and a factor greater than 1.0 sharpens the image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Alters the sharpness of the image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.sharpen(
            image,
            factor=self.factor,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class ShufflePixels(BaseTransform):
    def __init__(self, factor: float = 1.0, seed: int = 10, p: float = 1.0):
        """
        @param factor: a control parameter between 0.0 and 1.0. While a factor of
            0.0 returns the original image, a factor of 1.0 performs full shuffling

        @param seed: seed for numpy random generator to select random pixels
            for shuffling

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor
        self.seed = seed

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Shuffles the pixels of an image with respect to the shuffling factor. The
        factor denotes percentage of pixels to be shuffled and randomly selected
        Note: The actual number of pixels will be less than the percentage given
        due to the probability of pixels staying in place in the course of shuffling

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.shuffle_pixels(
            image,
            factor=self.factor,
            seed=self.seed,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class VFlip(BaseTransform):
    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Vertically flips an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.vflip(
            image,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


"""
Random Transforms

These classes below are similar to the non-random transforms in the sense
where they can be used with the Compose operator, etc. However, instead of
specifying specific parameters for the augmentation, with these functions
you can specify a range (or a list) to randomly choose from instead.

Example:
 >>> image = Image.open("....")
 >>> blur_tsfm = RandomBlur(min_radius=2.0, max_radius=5.0, p=0.5)
 >>> blurred_image = blur_tsfm(image)
"""


class RandomAspectRatio(BaseRandomRangeTransform):
    def __init__(self, min_ratio: float = 0.5, max_ratio: float = 2.0, p: float = 1.0):
        """
        @param min_ratio: the lower value on the range of aspect ratio values to choose
            from, i.e. the width/height ratio

        @param max_ratio: the upper value on the range of aspect ratio values to choose
            from, i.e. the width/height ratio

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_ratio, max_ratio, p)

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that randomly changes the aspect ratio of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.change_aspect_ratio(
            image,
            ratio=self.chosen_value,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomBlur(BaseRandomRangeTransform):
    def __init__(
        self, min_radius: float = 0.0, max_radius: float = 10.0, p: float = 1.0
    ):
        """
        @param min_radius: the lower value on the range of blur values to choose
            from. The larger the radius, the blurrier the image

        @param max_radius: the upper value on the range of blur values to choose
            from. The larger the radius, the blurrier the image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_radius, max_radius, p)

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that randomly blurs an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.blur(
            image,
            radius=self.chosen_value,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomBrightness(BaseRandomRangeTransform):
    def __init__(
        self, min_factor: float = 0.0, max_factor: float = 2.0, p: float = 1.0
    ):
        """
        @param min_factor: the lower value on the range of brightness values to choose
            from. The lower the factor, the darker the image

        @param max_factor: the upper value on the range of brightness values to choose
            from. The higher the factor, the brighter the image

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_factor, max_factor, p)

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that randomly changes the brightness of an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.brightness(
            image,
            factor=self.chosen_value,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomEmojiOverlay(BaseTransform):
    def __init__(
        self,
        emoji_directory: str = utils.SMILEY_EMOJI_DIR,
        opacity: float = 1.0,
        emoji_size: float = 0.15,
        x_pos: float = 0.4,
        y_pos: float = 0.4,
        p: float = 1.0,
    ):
        """
        @param emoji_directory: iopath directory uri containing the emoji images

        @param opacity: the lower the opacity, the more transparent the overlaid emoji

        @param emoji_size: size of the emoji is emoji_size * height of the original image

        @param x_pos: position of emoji relative to the image width

        @param y_pos: position of emoji relative to the image height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.emoji_directory = emoji_directory
        self.emoji_paths = utils.pathmgr.ls(emoji_directory)
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that overlays a random emoji onto an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        emoji_path = random.choice(self.emoji_paths)
        return F.overlay_emoji(
            image,
            emoji_path=os.path.join(self.emoji_directory, emoji_path),
            opacity=self.opacity,
            emoji_size=self.emoji_size,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomPixelization(BaseRandomRangeTransform):
    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 1.0, p: float = 1.0):
        """
        @param min_ratio: the lower value on the range of pixelization ratio values to
            choose from. Smaller values result in a more pixelated image, 1.0 indicates
            no change, and any value above one doesn't have a noticeable effect

        @param max_ratio: the upper value on the range of pixelization ratio values to
            choose from. Smaller values result in a more pixelated image, 1.0 indicates
            no change, and any value above one doesn't have a noticeable effect

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_ratio, max_ratio, p)

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that randomly pixelizes an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.pixelization(
            image,
            ratio=self.chosen_value,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )


class RandomRotation(BaseRandomRangeTransform):
    def __init__(
        self, min_degrees: float = 0.0, max_degrees: float = 180.0, p: float = 1.0
    ):
        """
        @param min_degrees: the lower value on the range of degree values to choose from

        @param max_degrees: the upper value on the range of degree values to choose from

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_degrees, max_degrees, p)

    def apply_random_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        Transform that randomly rotates an image

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return F.rotate(
            image,
            degrees=self.chosen_value,
            metadata=metadata,
            bboxes=bboxes,
            bbox_format=bbox_format,
        )
