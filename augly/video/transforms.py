#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from augly import audio as audaugs, utils
from augly.video import functional as F
from augly.video.augmenters import ffmpeg as af
from augly.video.helpers import identity_function


"""
Base Classes for Transforms
"""


class VidAugBaseClass:
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(self, *args, **kwargs) -> Any:
        """
        This function is to be implemented in the child classes.
        From this function, call the transform to be applied
        """
        raise NotImplementedError()


class BaseTransform(VidAugBaseClass):
    def __call__(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        force: bool = False,
        seed: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param force: if set to True, the transform will be applied. Otherwise,
            application is determined by the probability set

        @param seed: if provided, the random seed will be set to this before calling
            the transform

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        assert type(force) == bool, "Expected type bool for variable `force`"

        if not force and random.random() > self.p:
            return video_path

        if seed is not None:
            random.seed(seed)

        return self.apply_transform(video_path, output_path or video_path, metadata)

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
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
        self.min_val = min_val
        self.max_val = max_val
        self.chosen_value = None
        super().__init__(p)

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        self.chosen_value = (
            random.random() * (self.max_val - self.min_val)
        ) + self.min_val
        return self.apply_random_transform(video_path, output_path, metadata)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
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
 >>> blur_tsfm = Blur(radius=5.0, p=0.5)
 >>> blur_tsfm(video_path, output_path)
"""


class AddNoise(BaseTransform):
    def __init__(self, level: int = 25, p: float = 1.0):
        """
        @param level: noise strength for specific pixel component. Default value
            is 25. Allowed range is [0, 100], where 0 indicates no change

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.level = level

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Adds noise to a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.add_noise(video_path, output_path, self.level, metadata=metadata)


class ApplyLambda(BaseTransform):
    def __init__(
        self,
        aug_function: Callable[..., str] = identity_function,
        p: float = 1.0,
        **kwargs,
    ):
        """
        @param aug_function: the augmentation function to be applied onto the video
            (should expect a video path and output path as input and output the
            augmented video to the output path, then return the output path)

        @param p: the probability of the transform being applied; default value is 1.0

        @param **kwargs: the input attributes to be passed into `aug_function`
        """
        super().__init__(p)
        assert callable(aug_function), (
            repr(type(aug_function).__name__) + " object is not callable"
        )
        self.aug_function = aug_function
        self.kwargs = kwargs

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Apply a user-defined lambda on a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return self.aug_function(
            video_path, output_path, metadata=metadata, **self.kwargs
        )


class AudioSwap(BaseTransform):
    def __init__(self, audio_path: str, offset: float = 0.0, p: float = 1.0):
        """
        @param audio_path: the iopath uri to the audio you'd like to swap with the
            video's audio

        @param offset: starting point in seconds such that an audio clip of offset to
            offset + video_duration is used in the audio swap. Default value is zero

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.audio_path = audio_path
        self.offset = offset

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Swaps the video audio for the audio passed in provided an offset

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.audio_swap(
            video_path, self.audio_path, output_path, self.offset, metadata=metadata
        )


class AugmentAudio(BaseTransform):
    def __init__(
        self,
        audio_aug_function: Callable[
            ..., Tuple[np.ndarray, int]
        ] = audaugs.apply_lambda,
        p: float = 1.0,
        **audio_aug_kwargs,
    ):
        """
        @param audio_aug_function: the augmentation function to be applied onto the
            video's audio track. Should have the standard API of an AugLy audio
            augmentation, i.e. expect input audio as a numpy array or path & output
            path as input, and output the augmented audio to the output path

        @param p: the probability of the transform being applied; default value is 1.0

        @param audio_aug_kwargs: the input attributes to be passed into `audio_aug`
        """
        super().__init__(p)
        self.audio_aug_function = audio_aug_function
        self.audio_aug_kwargs = audio_aug_kwargs

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Augments the audio track of the input video using a given AugLy audio
        augmentation

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or
            returned

        @returns: the path to the augmented video
        """
        return F.augment_audio(
            video_path=video_path,
            audio_aug_function=self.audio_aug_function,
            output_path=output_path,
            metadata=metadata,
            **self.audio_aug_kwargs,
        )


class BlendVideos(BaseTransform):
    def __init__(
        self,
        overlay_path: str,
        opacity: float = 0.5,
        overlay_size: float = 1.0,
        x_pos: float = 0.0,
        y_pos: float = 0.0,
        use_second_audio: bool = True,
        p: float = 1.0,
    ):
        """
        @param overlay_path: the path to the video that will be overlaid onto the
            background video

        @param opacity: the lower the opacity, the more transparent the overlaid video

        @param overlay_size: size of the overlaid video is overlay_size * height of the
            background video

        @param x_pos: position of overlaid video relative to the background video width

        @param y_pos: position of overlaid video relative to the background video height

        @param use_second_audio: use the audio of the overlaid video rather than the
            audio of the background video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.overlay_path = overlay_path
        self.opacity = opacity
        self.overlay_size = overlay_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.use_second_audio = use_second_audio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays a video onto another video at position (width * x_pos, height * y_pos)
        at a lower opacity

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.blend_videos(
            video_path,
            self.overlay_path,
            output_path,
            self.opacity,
            self.overlay_size,
            self.x_pos,
            self.y_pos,
            self.use_second_audio,
            metadata=metadata,
        )


class Blur(BaseTransform):
    def __init__(self, sigma: float = 1.0, p: float = 1.0):
        """
        @param sigma: horizontal sigma, standard deviation of Gaussian blur

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.sigma = sigma

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Blurs a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.blur(video_path, output_path, self.sigma, metadata=metadata)


class Brightness(BaseTransform):
    def __init__(self, level: float = 0.15, p: float = 1.0):
        """
        @param level: the value must be a float value in range -1.0 to 1.0, where a
            negative value darkens and positive brightens

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.level = level

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Brightens or darkens a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.brightness(
            video_path,
            output_path,
            level=self.level,
            metadata=metadata,
        )


class ChangeAspectRatio(BaseTransform):
    def __init__(self, ratio: float = 1.0, p: float = 1.0):
        """
        @param ratio: aspect ratio, i.e. width/height, of the new video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.ratio = ratio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Changes the aspect ratio of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.change_aspect_ratio(
            video_path, output_path, self.ratio, metadata=metadata
        )


class ChangeVideoSpeed(BaseTransform):
    def __init__(self, factor: float = 1.0, p: float = 1.0):
        """
        @param factor: the factor by which to alter the speed of the video. A factor
            less than one will slow down the video, a factor equal to one won't alter
            the video, and a factor greater than one will speed up the video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Changes the speed of the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.change_video_speed(
            video_path, output_path, self.factor, metadata=metadata
        )


class ColorJitter(BaseTransform):
    def __init__(
        self,
        brightness_factor: float = 0,
        contrast_factor: float = 1.0,
        saturation_factor: float = 1.0,
        p: float = 1.0,
    ):
        """
        @param brightness_factor: set the brightness expression. The value must be
            a float value in range -1.0 to 1.0. The default value is 0

        @param contrast_factor: set the contrast expression. The value must be a
            float value in range -1000.0 to 1000.0. The default value is 1

        @param saturation_factor: set the saturation expression. The value must be a
            float in range 0.0 to 3.0. The default value is 1

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Color jitters the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.color_jitter(
            video_path,
            output_path,
            self.brightness_factor,
            self.contrast_factor,
            self.saturation_factor,
            metadata=metadata,
        )


class Concat(BaseTransform):
    def __init__(
        self,
        other_video_paths: List[str],
        src_video_path_index: int = 0,
        transition: Optional[af.TransitionConfig] = None,
        p: float = 1.0,
    ):
        """
        @param other_video_paths: a list of paths to the videos to be concatenated (in
            order) with the given video_path when called (which will be inserted in with
            this list of video paths at index src_video_path_index)

        @param src_video_path_index: for metadata purposes, this indicates which video in
            the list `video_paths` should be considered the `source` or original video

        @param transition: optional transition config between the clips

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.other_video_paths = other_video_paths
        self.src_video_path_index = src_video_path_index
        self.transition = transition

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Concatenates videos together. Resizes all other videos to the size of the
        `source` video (video_paths[src_video_path_index]), and modifies the sample
        aspect ratios to match (ffmpeg will fail to concat if SARs don't match)

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        video_paths = (
            self.other_video_paths[: self.src_video_path_index]
            + [video_path]
            + self.other_video_paths[self.src_video_path_index :]
        )
        return F.concat(
            video_paths,
            output_path,
            self.src_video_path_index,
            transition=self.transition,
            metadata=metadata,
        )


class Contrast(BaseTransform):
    def __init__(self, level: float = 1.0, p: float = 1.0):
        """
        @param level: the value must be a float value in range -1000.0 to 1000.0,
            where a negative value removes contrast and a positive value adds contrast

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.level = level

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Alters the contrast of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.contrast(video_path, output_path, self.level, metadata=metadata)


class Crop(BaseTransform):
    def __init__(
        self,
        left: float = 0.25,
        top: float = 0.25,
        right: float = 0.75,
        bottom: float = 0.75,
        p: float = 1.0,
    ):
        """
        @param left: left positioning of the crop; between 0 and 1, relative to
            the video width

        @param top: top positioning of the crop; between 0 and 1, relative to
            the video height

        @param right: right positioning of the crop; between 0 and 1, relative to
            the video width

        @param bottom: bottom positioning of the crop; between 0 and 1, relative to
            the video height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.left, self.right, self.top, self.bottom = left, right, top, bottom

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Crops the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.crop(
            video_path,
            output_path,
            self.left,
            self.top,
            self.right,
            self.bottom,
            metadata=metadata,
        )


class EncodingQuality(BaseTransform):
    def __init__(self, quality: int = 23, p: float = 1.0):
        """
        @param quality: CRF scale is 0–51, where 0 is lossless, 23 is the default,
            and 51 is worst quality possible. A lower value generally leads to higher
            quality, and a subjectively sane range is 17–28

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.quality = quality

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Alters the encoding quality of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.encoding_quality(
            video_path,
            output_path,
            quality=int(self.quality),
            metadata=metadata,
        )


class FPS(BaseTransform):
    def __init__(self, fps: int = 15, p: float = 1.0):
        """
        @param fps: the desired output frame rate. Note that a FPS value greater than
            the original FPS of the video will result in an unaltered video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.fps = fps

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Alters the FPS of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.fps(video_path, output_path, self.fps, metadata=metadata)


class Grayscale(BaseTransform):
    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Changes a video to be grayscale

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.grayscale(video_path, output_path, metadata=metadata)


class HFlip(BaseTransform):
    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Horizontally flips a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.hflip(video_path, output_path, metadata=metadata)


class HStack(BaseTransform):
    def __init__(
        self,
        second_video_path: str,
        use_second_audio: bool = False,
        p: float = 1.0,
    ):
        """
        @param second_video_path: the path to the video that will be stacked
            to the right

        @param use_second_audio: if set to True, the audio of the right video will
            be used instead of the left's

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.second_video_path = second_video_path
        self.use_second_audio = use_second_audio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Horizontally stacks two videos

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.hstack(
            video_path,
            self.second_video_path,
            output_path,
            self.use_second_audio,
            metadata=metadata,
        )


class InsertInBackground(BaseTransform):
    def __init__(
        self,
        background_path: Optional[str] = None,
        offset_factor: float = 0.0,
        transition: Optional[af.TransitionConfig] = None,
        p: float = 1.0,
    ):
        """
        @param background_path: the path to the video in which to insert the main
            video. If set to None, the main video will play in the middle of a silent
            video with black frames

        @param offset_factor: the point in the background video in which the main video
            starts to play (this factor is multiplied by the background video duration
            to determine the start point)

        @param transition: optional transition config between the clips

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.background_path = background_path
        self.offset_factor = offset_factor
        self.transition = transition

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Puts the video in the middle of the background video
        (at offset_factor * background.duration)

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.insert_in_background(
            video_path,
            output_path,
            self.background_path,
            self.offset_factor,
            transition=self.transition,
            metadata=metadata,
        )


class InsertInBackgroundMultiple(BaseTransform):
    def __init__(
        self,
        background_path: str,
        additional_video_paths: List[str],
        src_ids: List[str],
        seed: Optional[int] = None,
    ):
        """
        @param background_path: the path of the video in which to insert
            the main (and additional) video.

        @param additional_video_paths: list of additional video paths to
            be inserted alongside the main video; one clip from each of the
            input videos will be inserted in order.

        @param src_ids: the list of identifiers for the main video and additional videos.

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs.
        """
        super().__init__()
        self.background_path = background_path
        self.additional_video_paths = additional_video_paths
        self.src_ids = src_ids
        self.seed = seed

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Places the video (and the additional videos) in the middle of the background video.

        @param video_path: the path of the main video to be augmented.

        @param output_path: the path in which the output video will be stored.

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned
        """
        return F.insert_in_background_multiple(
            video_path=video_path,
            output_path=output_path,
            background_path=self.background_path,
            src_ids=self.src_ids,
            additional_video_paths=self.additional_video_paths,
            seed=self.seed,
            metadata=metadata,
        )


class Loop(BaseTransform):
    def __init__(self, num_loops: int = 0, p: float = 1.0):
        """
        @param num_loops: the number of times to loop the video. 0 means that the
            video will play once (i.e. no loops)

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.num_loops = num_loops

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Loops a video `num_loops` times

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.loop(video_path, output_path, self.num_loops, metadata=metadata)


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
        self.text, self.font_file = text, font_file
        self.text_color, self.opacity = text_color, opacity
        self.meme_bg_color, self.caption_height = meme_bg_color, caption_height

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Creates a new video that looks like a meme, given text and video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.meme_format(
            video_path,
            output_path,
            self.text,
            self.font_file,
            self.opacity,
            self.text_color,
            self.caption_height,
            self.meme_bg_color,
            metadata=metadata,
        )


class Overlay(BaseTransform):
    def __init__(
        self,
        overlay_path: str,
        overlay_size: Optional[float] = None,
        x_factor: float = 0.0,
        y_factor: float = 0.0,
        use_overlay_audio: bool = False,
        metadata: Optional[List[Dict[str, Any]]] = None,
        p: float = 1.0,
    ):
        """
        @param overlay_path: the path to the media (image or video) that will be
            overlaid onto the video

        @param overlay_size: size of the overlaid media with respect to the background
            video. If set to None, the original size of the overlaid media is used

        @param x_factor: specifies where the left side of the overlaid media should be
            placed, relative to the video width

        @param y_factor: specifies where the top side of the overlaid media should be
            placed, relative to the video height

        @param use_overlay_audio: if set to True and the media type is a video, the
            audio of the overlaid video will be used instead of the main/background
            video's audio

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.overlay_path = overlay_path
        self.overlay_size = overlay_size
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.use_overlay_audio = use_overlay_audio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays media onto the video at position (width * x_factor, height * y_factor)

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay(
            video_path,
            self.overlay_path,
            output_path,
            self.overlay_size,
            self.x_factor,
            self.y_factor,
            self.use_overlay_audio,
            metadata=metadata,
        )


class OverlayDots(BaseTransform):
    def __init__(
        self,
        num_dots: int = 100,
        dot_type: str = "colored",
        random_movement: bool = True,
        metadata: Optional[List[Dict[str, Any]]] = None,
        p: float = 1.0,
    ):
        """
        @param num_dots: the number of dots to add to each frame

        @param dot_type: specify if you would like "blur" or "colored"

        @param random_movement: whether or not you want the dots to randomly move
            around across the frame or to move across in a "linear" way

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.num_dots = num_dots
        self.dot_type = dot_type
        self.random_movement = random_movement

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays dots onto a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_dots(
            video_path,
            output_path,
            self.num_dots,
            self.dot_type,
            self.random_movement,
            metadata=metadata,
        )


class OverlayEmoji(BaseTransform):
    def __init__(
        self,
        emoji_path: str = utils.EMOJI_PATH,
        x_factor: float = 0.4,
        y_factor: float = 0.4,
        opacity: float = 1.0,
        emoji_size: float = 0.15,
        p: float = 1.0,
    ):
        """
        @param emoji_path: iopath uri to the emoji image

        @param x_factor: specifies where the left side of the emoji should be
            placed, relative to the video width

        @param y_factor: specifies where the top side of the emoji should be placed,
            relative to the video height

        @param opacity: opacity of the emoji image

        @param emoji_size: emoji size relative to the height of the video frame

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.emoji_path = emoji_path
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.opacity = opacity
        self.emoji_size = emoji_size

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays an emoji onto each frame of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_emoji(
            video_path,
            output_path,
            self.emoji_path,
            self.x_factor,
            self.y_factor,
            self.opacity,
            self.emoji_size,
            metadata=metadata,
        )


class OverlayOntoBackgroundVideo(BaseTransform):
    def __init__(
        self,
        background_path: str,
        overlay_size: Optional[float] = 0.7,
        x_factor: float = 0.0,
        y_factor: float = 0.0,
        use_background_audio: bool = False,
        p: float = 1.0,
    ):
        """
        @param background_path: the path to the background video

        @param overlay_size: size of the overlaid media with respect to the background
            video. If set to None, the original size of the overlaid media is used

        @param x_factor: specifies where the left side of the overlaid media should be
            placed, relative to the video width

        @param y_factor: specifies where the top side of the overlaid media should be
            placed, relative to the video height

        @param use_background_audio: if set to True and the media type is a video, the
            audio of the background video will be used instead of the src video's audio

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.background_path = background_path
        self.overlay_size = overlay_size
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.use_background_audio = use_background_audio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays the video onto a background video, pointed to by background_path

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_onto_background_video(
            video_path,
            self.background_path,
            output_path,
            self.overlay_size,
            self.x_factor,
            self.y_factor,
            self.use_background_audio,
            metadata=metadata,
        )


class OverlayOntoScreenshot(BaseTransform):
    def __init__(
        self,
        template_filepath: str = utils.TEMPLATE_PATH,
        template_bboxes_filepath: str = utils.BBOXES_PATH,
        max_image_size_pixels: Optional[int] = None,
        crop_src_to_fit: bool = False,
        p: float = 1.0,
    ):
        """
        @param template_filepath: iopath uri to the screenshot template

        @param template_bboxes_filepath: iopath uri to the file containing the bounding
            box for each template

        @param max_image_size_pixels: if provided, the template image and/or src video
            will be scaled down to avoid an output image with an area greater than this
            size (in pixels)

        @param crop_src_to_fit: if True, the src image will be cropped if necessary to
            fit into the template image. If False, the src image will instead be resized
            if necessary

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.template_filepath = template_filepath
        self.template_bboxes_filepath = template_bboxes_filepath
        self.max_image_size_pixels = max_image_size_pixels
        self.crop_src_to_fit = crop_src_to_fit

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays the video onto a screenshot template so it looks like it was
        screen-recorded on Instagram

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_onto_screenshot(
            video_path,
            output_path,
            self.template_filepath,
            self.template_bboxes_filepath,
            self.max_image_size_pixels,
            self.crop_src_to_fit,
            metadata=metadata,
        )


class OverlayShapes(BaseTransform):
    def __init__(
        self,
        num_shapes: int = 1,
        shape_type: Optional[str] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: Optional[int] = None,
        radius: Optional[float] = None,
        random_movement: bool = True,
        topleft: Optional[Tuple[float, float]] = None,
        bottomright: Optional[Tuple[float, float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        p: float = 1.0,
    ):
        """
        @param num_shapes: the number of shapes to add to each frame

        @param shape_type: specify if you would like circles or rectangles

        @param colors: list of (R, G, B) colors to sample from

        @param thickness: specify the thickness desired for the shapes

        @param random_movement: whether or not you want the shapes to randomly move
            around across the frame or to move across in a "linear" way

        @param topleft: specifying the boundary of shape region. The boundary are all
            floats [0, 1] representing the fraction w.r.t width/height

        @param bottomright: specifying the boundary of shape region. The boundary are
            all floats [0, 1] epresenting the fraction w.r.t width/height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.num_shapes = num_shapes
        self.shape_type = shape_type
        self.colors = colors
        self.thickness = thickness
        self.radius = radius
        self.random_movement = random_movement
        self.topleft = topleft
        self.bottomright = bottomright

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays random shapes onto a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_shapes(
            video_path,
            output_path,
            self.num_shapes,
            self.shape_type,
            self.colors,
            self.thickness,
            self.radius,
            self.random_movement,
            self.topleft,
            self.bottomright,
            metadata=metadata,
        )


class OverlayText(BaseTransform):
    def __init__(
        self,
        text_len: int = 10,
        text_change_nth: Optional[int] = None,
        fonts: Optional[List[Tuple[Any, Optional[str]]]] = None,
        fontscales: Optional[Tuple[float, float]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: Optional[int] = None,
        random_movement: bool = False,
        topleft: Optional[Tuple[float, float]] = None,
        bottomright: Optional[Tuple[float, float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        p: float = 1.0,
    ):
        """
        @param text_len: length of string for randomized texts.

        @param text_change_nth: change random text every nth frame. None means using
            same text for all frames

        @param fonts: list of fonts to sample from. Each font can be a cv2 fontFace,
            a PIL ImageFont, or a path to a PIL ImageFont file. Each font is coupled
            with a chars file (the second item in the tuple) - a path to a file which
            contains the characters associated with the given font. For example, non-
            western alphabets have different valid characters than the roman alphabet,
            and these must be specified in order to construct random valid text in
            that font. If the chars file path is None, the roman alphabet will be used

        @param fontscales: 2-tuple of float (min_scale, max_scale)

        @param colors: list of (R, G, B) colors to sample from

        @param thickness: specify the thickness desired for the text

        @param random_movement: whether or not you want the text to randomly move around
            across frame or to move across in a "linear" way

        @param topleft: specifying the boundary of text region. The boundary are all
            floats [0, 1] representing the fraction w.r.t width/height

        @param bottomright: specifying the boundary of text region. The boundary are
            all floats [0, 1] representing the fraction w.r.t width/height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.text_len = text_len
        self.text_change_nth = text_change_nth
        self.fonts = fonts
        self.fontscales = fontscales
        self.colors = colors
        self.thickness = thickness
        self.random_movement = random_movement
        self.topleft = topleft
        self.bottomright = bottomright

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Overlays random text onto a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.overlay_text(
            video_path,
            output_path,
            self.text_len,
            self.text_change_nth,
            self.fonts,
            self.fontscales,
            self.colors,
            self.thickness,
            self.random_movement,
            self.topleft,
            self.bottomright,
            metadata=metadata,
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
        @param w_factor: pad right and left with w_factor * frame width

        @param h_factor: pad bottom and top with h_factor * frame height

        @param color: RGB color of the padded margin

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.w_factor, self.h_factor = w_factor, h_factor
        self.color = color

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Pads the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.pad(
            video_path,
            output_path,
            self.w_factor,
            self.h_factor,
            self.color,
            metadata=metadata,
        )


class PerspectiveTransformAndShake(BaseTransform):
    def __init__(
        self,
        sigma: float = 50.0,
        shake_radius: float = 0.0,
        seed: Optional[int] = None,
        p: float = 1.0,
    ):
        """
        @param sigma: the standard deviation of the distribution of destination
            coordinates. the larger the sigma value, the more intense the transform

        @param shake_radius: determines the amount by which to "shake" the video;
            the larger the radius, the more intense the shake

        @param seed: if provided, this will set the random seed to ensure consistency
            between runs

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.sigma, self.shake_radius = sigma, shake_radius
        self.seed = seed

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Apply a perspective transform to the video so it looks like it was taken
        as a photo from another device (e.g. taking a video from your phone of a
        video on a computer). Also has a shake factor to mimic the shakiness of
        someone holding a phone

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.perspective_transform_and_shake(
            video_path, output_path, self.sigma, self.shake_radius, self.seed, metadata
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
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Pixelizes the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.pixelization(
            video_path,
            output_path,
            ratio=self.ratio,
            metadata=metadata,
        )


class RemoveAudio(BaseTransform):
    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Removes the audio stream from a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.remove_audio(video_path, output_path, metadata=metadata)


class ReplaceWithBackground(BaseTransform):
    def __init__(
        self,
        background_path: Optional[str] = None,
        source_offset: float = 0.0,
        background_offset: float = 0.0,
        source_percentage: float = 0.5,
        transition: Optional[af.TransitionConfig] = None,
        p: float = 1.0,
    ):
        """
        @param background_path: the path to the video in which to insert the main
            video. If set to None, the main video will play in the middle of a silent
            video with black frames

        @param offset_factor: the point in the background video in which the main video
            starts to play (this factor is multiplied by the background video duration
            to determine the start point)

        @param transition: optional transition config between the clips

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.background_path = background_path
        self.source_offset = source_offset
        self.background_offset = background_offset
        self.source_percentage = source_percentage
        self.transition = transition

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Puts the video in the middle of the background video
        (at offset_factor * background.duration)

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.replace_with_background(
            video_path,
            output_path,
            background_path=self.background_path,
            source_offset=self.source_offset,
            background_offset=self.background_offset,
            source_percentage=self.source_percentage,
            transition=self.transition,
            metadata=metadata,
        )


class ReplaceWithColorFrames(BaseTransform):
    def __init__(
        self,
        offset_factor: float = 0.0,
        duration_factor: float = 1.0,
        color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
        transition: Optional[af.TransitionConfig] = None,
        p: float = 1.0,
    ):
        """
        @param offset_factor: start point of the replacement relative to the video
            duration (this parameter is multiplied by the video duration)

        @param duration_factor: the length of the replacement relative to the video
            duration (this parameter is multiplied by the video duration)

        @param color: RGB color of the replaced frames. Default color is black

        @param transition: optional transition config between the clips

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.offset_factor, self.duration_factor = offset_factor, duration_factor
        self.color = color
        self.transition = transition

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Replaces part of the video with frames of the specified color

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.replace_with_color_frames(
            video_path,
            output_path,
            self.offset_factor,
            self.duration_factor,
            self.color,
            transition=self.transition,
            metadata=metadata,
        )


class Resize(BaseTransform):
    def __init__(
        self, height: Optional[int] = None, width: Optional[int] = None, p: float = 1.0
    ):
        """
        @param height: the height in which the video should be resized to. If None,
            the original video height will be used

        @param width: the width in which the video should be resized to. If None, the
            original video width will be used

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.height, self.width = height, width

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Resizes a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.resize(
            video_path, output_path, self.height, self.width, metadata=metadata
        )


class Rotate(BaseTransform):
    def __init__(self, degrees: float = 15, p: float = 1.0):
        """
        @param degrees: expression for the angle by which to rotate the input video
            clockwise, expressed in degrees (supports negative values as well)

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.degrees = degrees

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Rotates a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.rotate(video_path, output_path, self.degrees, metadata=metadata)


class Scale(BaseTransform):
    def __init__(self, factor: float = 0.5, p: float = 1.0):
        """
        @param factor: the ratio by which the video should be down-scaled or upscaled

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Alters the resolution of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.scale(video_path, output_path, self.factor, metadata=metadata)


class Shift(BaseTransform):
    def __init__(
        self,
        x_factor: float = 0.0,
        y_factor: float = 0.0,
        color: Tuple[int, int, int] = utils.DEFAULT_COLOR,
        p: float = 1.0,
    ):
        """
        @param x_factor: the horizontal amount that the video should be shifted,
            relative to the width of the video

        @param y_factor: the vertical amount that the video should be shifted,
            relative to the height of the video

        @param color: RGB color of the margin generated by the shift. Default color
            is black

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.x_factor, self.y_factor = x_factor, y_factor
        self.color = color

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Shifts the original frame position from the center by a vector
        (width * x_factor, height * y_factor) and pads the rest with a
        colored margin

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.shift(
            video_path,
            output_path,
            self.x_factor,
            self.y_factor,
            self.color,
            metadata=metadata,
        )


class TimeCrop(BaseTransform):
    def __init__(
        self, offset_factor: float = 0.0, duration_factor: float = 1.0, p: float = 1.0
    ):
        """
        @param offset_factor: start point of the crop relative to the video duration
            (this parameter is multiplied by the video duration)

        @param duration_factor: the length of the crop relative to the video duration
            (this parameter is multiplied by the video duration)

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.offset_factor, self.duration_factor = offset_factor, duration_factor

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Crops the video using the specified offset and duration factors

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.time_crop(
            video_path,
            output_path,
            self.offset_factor,
            self.duration_factor,
            metadata=metadata,
        )


class TimeDecimate(BaseTransform):
    def __init__(
        self,
        start_offset_factor: float = 0.0,
        on_factor: float = 0.2,
        off_factor: float = 0.5,
        transition: Optional[af.TransitionConfig] = None,
        p: float = 1.0,
    ):
        """
        @param start_offset_factor: relative to the video duration; the offset
            at which to start taking "on" segments

        @param on_factor: relative to the video duration; the amount of time each
            "on" video chunk should be

        @param off_factor: relative to the "on" duration; the amount of time each
            "off" video chunk should be

        @param transition: optional transition config between the clips

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.start_offset_factor = start_offset_factor
        self.on_factor, self.off_factor = on_factor, off_factor
        self.transition = transition

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Removes evenly sized (off) chunks, and concatenates evenly spaced (on)
        chunks from the video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.time_decimate(
            video_path,
            output_path,
            start_offset_factor=self.start_offset_factor,
            on_factor=self.on_factor,
            off_factor=self.off_factor,
            transition=self.transition,
            metadata=metadata,
        )


class Trim(BaseTransform):
    def __init__(
        self, start: Optional[float] = None, end: Optional[float] = None, p: float = 1.0
    ):
        """
        @param start: starting point in seconds of when the trimmed video should start.
            If None, start will be 0

        @param end: ending point in seconds of when the trimmed video should end.
            If None, the end will be the duration of the video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.start, self.end = start, end

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Trims the video using the specified start and end parameters

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.trim(video_path, output_path, self.start, self.end, metadata=metadata)


class VFlip(BaseTransform):
    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Vertically flips a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.vflip(video_path, output_path, metadata=metadata)


class VStack(BaseTransform):
    def __init__(
        self,
        second_video_path: str,
        use_second_audio: bool = False,
        p: float = 1.0,
    ):
        """
        @param second_video_path: the path to the video that will be stacked on
            the bottom

        @param use_second_audio: if set to True, the audio of the bottom video will
            be used instead of the top's

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.second_video_path = second_video_path
        self.use_second_audio = use_second_audio

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Vertically stacks two videos

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.vstack(
            video_path,
            self.second_video_path,
            output_path,
            self.use_second_audio,
            metadata=metadata,
        )


"""
Random Transforms

These classes below are similar to the non-random transforms in the sense
where they can be used with the Compose operator, etc. However, instead of
specifying specific parameters for the augmentation, with these functions
you can specify a range (or a list) to randomly choose from instead.

Example:
 >>> blur_tsfm = RandomBlur(min_radius=2.0, max_radius=5.0, p=0.5)
 >>> blur_tsfm(video_path, output_path)
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
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the aspect ratio of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.change_aspect_ratio(
            video_path, output_path, ratio=self.chosen_value, metadata=metadata
        )


class RandomBlur(BaseRandomRangeTransform):
    def __init__(self, min_sigma: float = 0.0, max_sigma: float = 10.0, p: float = 1.0):
        """
        @param min_sigma: the lower value on the range of blur values to choose from.
            The larger the radius, the blurrier the video

        @param max_sigma: the upper value on the range of blur values to choose from.
            The larger the radius, the blurrier the video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_sigma, max_sigma, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly blurs a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.blur(
            video_path, output_path, sigma=self.chosen_value, metadata=metadata
        )


class RandomBrightness(BaseRandomRangeTransform):
    def __init__(self, min_level: float = -1.0, max_level: float = 1.0, p: float = 1.0):
        """
        @param min_level: the lower value on the range of brightness values to choose
            from. The lower the factor, the darker the video

        @param max_level: the upper value on the range of brightness values to choose
            from. The higher the factor, the brighter the video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_level, max_level, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the brightness of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.brightness(
            video_path,
            output_path,
            level=self.chosen_value,
            metadata=metadata,
        )


class RandomContrast(BaseRandomRangeTransform):
    def __init__(
        self, min_factor: float = -5.0, max_factor: float = 5.0, p: float = 1.0
    ):
        """
        @param min_factor: the lower value on the range of contrast values to choose
            from. The lower the factor, the less contrast

        @param max_factor: the upper value on the range of contrast values to choose
            from. The higher the factor, the more contrast

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_factor, max_factor, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the contrast of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.contrast(
            video_path, output_path, level=self.chosen_value, metadata=metadata
        )


class RandomEmojiOverlay(BaseTransform):
    def __init__(
        self,
        emoji_directory: str = utils.SMILEY_EMOJI_DIR,
        opacity: float = 1.0,
        emoji_size: float = 0.15,
        x_factor: float = 0.4,
        y_factor: float = 0.4,
        p: float = 1.0,
    ):
        """
        @param emoji_directory: iopath directory uri containing the emoji images

        @param opacity: the lower the opacity, the more transparent the overlaid emoji

        @param emoji_size: size of the emoji is emoji_size * height of the
            original video

        @param x_factor: position of emoji relative to the video width

        @param y_factor: position of emoji relative to the video height

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.emoji_directory = emoji_directory
        self.emoji_paths = utils.pathmgr.ls(emoji_directory)
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_factor = x_factor
        self.y_factor = y_factor

    def apply_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that overlays a random emoji onto a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        emoji_path = random.choice(self.emoji_paths)
        return F.overlay_emoji(
            video_path,
            output_path,
            emoji_path=os.path.join(self.emoji_directory, emoji_path),
            opacity=self.opacity,
            emoji_size=self.emoji_size,
            x_factor=self.x_factor,
            y_factor=self.y_factor,
            metadata=metadata,
        )


class RandomEncodingQuality(BaseRandomRangeTransform):
    def __init__(self, min_quality: int = 10, max_quality: int = 40, p: float = 1.0):
        """
        @param min_quality: the lower value on the range of encoding quality values
            to choose from. CRF scale is 0–51, where 0 is lossless, 23 is the default,
            and 51 is worst quality possible. a lower value generally leads to higher
            quality, and a subjectively sane range is 17–28

        @param max_quality: the upper value on the range of encoding quality values to
            choose from. CRF scale is 0–51, where 0 is lossless, 23 is the default, and
            51 is worst quality possible. a lower value generally leads to higher
            quality, and a subjectively sane range is 17–28

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_quality, max_quality, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the encoding quality of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.encoding_quality(
            video_path,
            output_path,
            quality=int(self.chosen_value),
            metadata=metadata,
        )


class RandomFPS(BaseRandomRangeTransform):
    def __init__(self, min_fps: float = 5.0, max_fps: float = 30.0, p: float = 1.0):
        """
        @param min_fps: the lower value on the range of fps values to choose from

        @param max_fps: the upper value on the range of fps values to choose from. Note
            that a FPS value greater than the original FPS of the video will result in
            an unaltered video

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_fps, max_fps, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the FPS of a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.fps(video_path, output_path, fps=self.chosen_value, metadata=metadata)


class RandomNoise(BaseRandomRangeTransform):
    def __init__(self, min_level: int = 0, max_level: int = 50, p: float = 1.0):
        """
        @param min_level: the lower value on the range of noise strength level
            values to choose from. 0 indicates no change, allowed range is [0, 100]

        @param max_level: the upper value on the range of noise strength level values
            to choose from. 0 indicates no change, allowed range is [0, 100]

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_level, max_level, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly adds noise to a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.add_noise(
            video_path, output_path, level=int(self.chosen_value), metadata=metadata
        )


class RandomPixelization(BaseRandomRangeTransform):
    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 1.0, p: float = 1.0):
        """
        @param min_ratio: the lower value on the range of pixelization ratio values to
            choose from. Smaller values result in a more pixelated video, 1.0 indicates
            no change, and any value above one doesn't have a noticeable effect

        @param max_ratio: the upper value on the range of pixelization ratio values to
            choose from. Smaller values result in a more pixelated video, 1.0 indicates
            no change, and any value above one doesn't have a noticeable effect

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_ratio, max_ratio, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly pixelizes a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.pixelization(
            video_path,
            output_path,
            ratio=self.chosen_value,
            metadata=metadata,
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
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly rotates a video

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.rotate(
            video_path, output_path, degrees=self.chosen_value, metadata=metadata
        )


class RandomVideoSpeed(BaseRandomRangeTransform):
    def __init__(
        self, min_factor: float = 0.25, max_factor: float = 4.0, p: float = 1.0
    ):
        """
        @param min_factor: the lower value on the range of speed values to choose
            from. A factor less than one will slow down the video, a factor equal to
            one won't alter the video, and a factor greater than one will speed up the
            video relative to the original speed

        @param max_factor: the upper value on the range of speed values to choose from.
            A factor less than one will slow down the video, a factor equal to one won't
            alter the video, and a factor greater than one will speed up the video
            relative to the original speed

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(min_factor, max_factor, p)

    def apply_random_transform(
        self,
        video_path: str,
        output_path: str,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Transform that randomly changes the video speed

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.
            If not passed in, the original video file will be overwritten

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, fps, etc. will be appended
            to the inputted list. If set to None, no metadata will be appended or returned

        @returns: the path to the augmented video
        """
        return F.change_video_speed(
            video_path, output_path, factor=self.chosen_value, metadata=metadata
        )
