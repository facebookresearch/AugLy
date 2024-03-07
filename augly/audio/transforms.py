#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from augly import utils
from augly.audio import functional as F
from augly.audio.utils import RNGSeed


"""
Base Classes for Transforms
"""


class BaseTransform:
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int = utils.DEFAULT_SAMPLE_RATE,
        metadata: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @param force: if set to True, the transform will be applied. otherwise,
            application is determined by the probability set

        @returns: the augmented audio array and sample rate
        """
        assert isinstance(audio, np.ndarray), "Audio passed in must be a np.ndarray"
        assert type(force) == bool, "Expected type bool for variable `force`"

        if not force and random.random() > self.p:
            return audio, sample_rate

        return self.apply_transform(audio, sample_rate, metadata)

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
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
 >>> audio_array = np.array([...])
 >>> pitch_shift_tsfm = PitchShift(n_steps=4.0, p=0.5)
 >>> shifted_audio = pitch_shift_tsfm(audio_array, sample_rate)
"""


class AddBackgroundNoise(BaseTransform):
    def __init__(
        self,
        background_audio: Optional[Union[str, np.ndarray]] = None,
        snr_level_db: float = 10.0,
        seed: Optional[RNGSeed] = None,
        p: float = 1.0,
    ):
        """
        @param background_audio: the path to the background audio or a variable of type
            np.ndarray containing the background audio. If set to `None`, the background
            audio will be white noise

        @param snr_level_db: signal-to-noise ratio in dB

        @param seed: a NumPy random generator (or seed) such that these results
            remain reproducible

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.background_audio = background_audio
        self.snr_level_db = snr_level_db
        self.seed = seed

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Mixes in a background sound into the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.add_background_noise(
            audio,
            sample_rate,
            self.background_audio,
            self.snr_level_db,
            self.seed,
            metadata=metadata,
        )


class ApplyLambda(BaseTransform):
    def __init__(
        self,
        aug_function: Callable[..., Tuple[np.ndarray, int]] = lambda x, y: (x, y),
        p: float = 1.0,
    ):
        """
        @param aug_function: the augmentation function to be applied onto the audio
            (should expect the audio np.ndarray & sample rate int as input, and return
            the transformed audio & sample rate)

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.aug_function = aug_function

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply a user-defined lambda to the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.apply_lambda(audio, sample_rate, self.aug_function, metadata=metadata)


class ChangeVolume(BaseTransform):
    def __init__(self, volume_db: float = 0.0, p: float = 1.0):
        """
        @param volume_db: the decibel amount by which to either increase (positive
            value) or decrease (negative value) the volume of the audio

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.volume_db = volume_db

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Changes the volume of the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.change_volume(audio, sample_rate, self.volume_db, metadata=metadata)


class Clicks(BaseTransform):
    def __init__(self, seconds_between_clicks: float = 0.5, p: float = 1.0):
        """
        @param seconds_between_clicks: the amount of time between each click that will
            be added to the audio, in seconds

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.seconds_between_clicks = seconds_between_clicks

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Adds clicks to the audio at a given regular interval

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.clicks(
            audio, sample_rate, self.seconds_between_clicks, metadata=metadata
        )


class Clip(BaseTransform):
    def __init__(
        self, offset_factor: float = 0.0, duration_factor: float = 1.0, p: float = 1.0
    ):
        """
        @param offset_factor: start point of the crop relative to the audio duration
            (this parameter is multiplied by the audio duration)

        @param duration_factor: the length of the crop relative to the audio duration
            (this parameter is multiplied by the audio duration)

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.offset_factor = offset_factor
        self.duration_factor = duration_factor

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Clips the audio using the specified offset and duration factors

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.clip(
            audio,
            sample_rate,
            self.offset_factor,
            self.duration_factor,
            metadata=metadata,
        )


class Harmonic(BaseTransform):
    def __init__(
        self,
        kernel_size: int = 31,
        power: float = 2.0,
        margin: float = 1.0,
        p: float = 1.0,
    ):
        """
        @param kernel_size: kernel size for the median filters

        @param power: exponent for the Wiener filter when constructing soft mask matrices

        @param margin: margin size for the masks

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.kernel_size = kernel_size
        self.power = power
        self.margin = margin

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Extracts the harmonic part of the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.harmonic(
            audio,
            sample_rate,
            self.kernel_size,
            self.power,
            self.margin,
            metadata=metadata,
        )


class HighPassFilter(BaseTransform):
    def __init__(self, cutoff_hz: float = 3000.0, p: float = 1.0):
        """
        @param cutoff_hz: frequency (in Hz) where signals with lower frequencies will
            begin to be reduced by 6dB per octave (doubling in frequency) below this point

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.cutoff_hz = cutoff_hz

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Allows audio signals with a frequency higher than the given cutoff to pass
        through and attenuates signals with frequencies lower than the cutoff frequency

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.high_pass_filter(audio, sample_rate, self.cutoff_hz, metadata=metadata)


class InsertInBackground(BaseTransform):
    def __init__(
        self,
        offset_factor: float = 0.0,
        background_audio: Optional[Union[str, np.ndarray]] = None,
        seed: Optional[RNGSeed] = None,
        p: float = 1.0,
    ):
        """
        @param offset_factor: start point of the crop relative to the background duration
            (this parameter is multiplied by the background duration)

        @param background_audio: the path to the background audio or a variable of type
            np.ndarray containing the background audio. If set to `None`, the background
            audio will be white noise

        @param seed: a NumPy random generator (or seed) such that these results
            remain reproducible

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.offset_factor = offset_factor
        self.background_audio = background_audio
        self.seed = seed

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Non-overlapping insert audio in a background audio.

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.insert_in_background(
            audio,
            sample_rate,
            self.offset_factor,
            self.background_audio,
            self.seed,
            metadata=metadata,
        )


class InvertChannels(BaseTransform):
    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Inverts the channels of the audio.

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.invert_channels(audio, sample_rate, metadata=metadata)


class Loop(BaseTransform):
    def __init__(self, n: int = 1, p: float = 1.0):
        """
        @param n: the number of times the audio will be looped

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.n = n

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Loops the audio 'n' times

        @param audio: the path to the audio or a variable of type np.ndarray that
            will be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.loop(audio, sample_rate, self.n, metadata=metadata)


class LowPassFilter(BaseTransform):
    def __init__(self, cutoff_hz: float = 500.0, p: float = 1.0):
        """
        @param cutoff_hz: frequency (in Hz) where signals with higher frequencies will
            begin to be reduced by 6dB per octave (doubling in frequency) above this point

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.cutoff_hz = cutoff_hz

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Allows audio signals with a frequency lower than the given cutoff to pass through
        and attenuates signals with frequencies higher than the cutoff frequency

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.low_pass_filter(audio, sample_rate, self.cutoff_hz, metadata=metadata)


class Normalize(BaseTransform):
    def __init__(
        self,
        norm: Optional[float] = np.inf,
        axis: int = 0,
        threshold: Optional[float] = None,
        fill: Optional[bool] = None,
        p: float = 1.0,
    ):
        """
        @param norm: the type of norm to compute:
            - np.inf: maximum absolute value
            - -np.inf: minimum absolute value
            - 0: number of non-zeros (the support)
            - float: corresponding l_p norm
            - None: no normalization is performed

        @param axis: axis along which to compute the norm

        @param threshold: if provided, only the columns (or rows) with norm of at
            least `threshold` are normalized

        @param fill: if None, then columns (or rows) with norm below `threshold` are
            left as is. If False, then columns (rows) with norm below `threshold` are
            set to 0. If True, then columns (rows) with norm below `threshold` are
            filled uniformly such that the corresponding norm is 1

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.norm, self.axis = norm, axis
        self.threshold, self.fill = threshold, fill

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Normalizes the audio array along the chosen axis (norm(audio, axis=axis) == 1)

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.normalize(
            audio,
            sample_rate,
            self.norm,
            self.axis,
            self.threshold,
            self.fill,
            metadata=metadata,
        )


class PeakingEqualizer(BaseTransform):
    def __init__(
        self,
        center_hz: float = 500.0,
        q: float = 1.0,
        gain_db: float = -3.0,
        p: float = 1.0,
    ):
        """
        @param center_hz: point in the frequency spectrum at which EQ is applied

        @param q: ratio of center frequency to bandwidth; bandwidth is inversely
            proportional to Q, meaning that as you raise Q, you narrow the bandwidth

        @param gain_db: amount of gain (boost) or reduction (cut) that is applied
            at a given frequency. Beware of clipping when using positive gain

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.center_hz = center_hz
        self.q = q
        self.gain_db = gain_db

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies a two-pole peaking equalization filter. The signal-level at and around
        `center_hz` can be increased or decreased, while all other frequencies are unchanged

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.peaking_equalizer(
            audio, sample_rate, self.center_hz, self.q, self.gain_db, metadata=metadata
        )


class Percussive(BaseTransform):
    def __init__(
        self,
        kernel_size: int = 31,
        power: float = 2.0,
        margin: float = 1.0,
        p: float = 1.0,
    ):
        """
        @param kernel_size: kernel size for the median filters

        @param power: exponent for the Wiener filter when constructing soft mask matrices

        @param margin: margin size for the masks

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.kernel_size = kernel_size
        self.power = power
        self.margin = margin

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Extracts the percussive part of the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.percussive(
            audio,
            sample_rate,
            self.kernel_size,
            self.power,
            self.margin,
            metadata=metadata,
        )


class PitchShift(BaseTransform):
    def __init__(self, n_steps: float = 1.0, p: float = 1.0):
        """
        @param n_steps: each step is equal to one semitone

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.n_steps = n_steps

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Shifts the pitch of the audio by `n_steps`

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.pitch_shift(audio, sample_rate, self.n_steps, metadata=metadata)


class Reverb(BaseTransform):
    def __init__(
        self,
        reverberance: float = 50.0,
        hf_damping: float = 50.0,
        room_scale: float = 100.0,
        stereo_depth: float = 100.0,
        pre_delay: float = 0.0,
        wet_gain: float = 0.0,
        wet_only: bool = False,
        p: float = 1.0,
    ):
        """
        @param reverberance: (%) sets the length of the reverberation tail. This
            determines how long the reverberation continues for after the original
            sound being reverbed comes to an end, and so simulates the "liveliness"
            of the room acoustics

        @param hf_damping: (%) increasing the damping produces a more "muted" effect.
            The reverberation does not build up as much, and the high frequencies decay
            faster than the low frequencies

        @param room_scale: (%) sets the size of the simulated room. A high value will
            simulate the reverberation effect of a large room and a low value will
            simulate the effect of a small room

        @param stereo_depth: (%) sets the apparent "width" of the reverb effect for
            stereo tracks only. Increasing this value applies more variation between
            left and right channels, creating a more "spacious" effect. When set at
            zero, the effect is applied independently to left and right channels

        @param pre_delay: (ms) delays the onset of the reverberation for the set time
            after the start of the original input. This also delays the onset of the
            reverb tail

        @param wet_gain: (db) applies volume adjustment to the reverberation ("wet")
            component in the mix

        @param wet_only: only the wet signal (added reverberation) will be in the
            resulting output, and the original audio will be removed

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.reverberance = reverberance
        self.hf_damping = hf_damping
        self.room_scale = room_scale
        self.stereo_depth = stereo_depth
        self.pre_delay = pre_delay
        self.wet_gain = wet_gain
        self.wet_only = wet_only

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Adds reverberation to the audio

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.reverb(
            audio,
            sample_rate,
            self.reverberance,
            self.hf_damping,
            self.room_scale,
            self.stereo_depth,
            self.pre_delay,
            self.wet_gain,
            self.wet_only,
            metadata=metadata,
        )


class Speed(BaseTransform):
    def __init__(self, factor: float = 2.0, p: float = 1.0):
        """
        @param factor: the speed factor. If rate > 1 the audio will be sped up by that
            factor; if rate < 1 the audio will be slowed down by that factor

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Changes the speed of the audio, affecting pitch as well

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.speed(audio, sample_rate, self.factor, metadata=metadata)


class Tempo(BaseTransform):
    def __init__(self, factor: float = 2.0, p: float = 1.0):
        """
        @param factor: the tempo factor. If rate > 1 the audio will be sped up by that
            factor; if rate < 1 the audio will be slowed down by that factor, without
            affecting the pitch

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.factor = factor

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Adjusts the tempo of the audio by a given factor

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.tempo(audio, sample_rate, self.factor, metadata=metadata)


class TimeStretch(BaseTransform):
    def __init__(self, rate: float = 1.5, p: float = 1.0):
        """
        @param rate: the time stretch factor

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.rate = rate

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Time-stretches the audio by a fixed rate

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.time_stretch(audio, sample_rate, self.rate, metadata=metadata)


class ToMono(BaseTransform):
    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Converts the audio from stereo to mono by averaging samples across channels

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.to_mono(audio, sample_rate, metadata=metadata)


class FFTConvolve(BaseTransform):
    def __init__(
        self,
        normalize: bool = True,
        impulse_audio: Optional[Union[str, np.ndarray]] = None,
        seed: Optional[RNGSeed] = None,
        p: float = 1.0,
    ):
        """
        @param normalize: if True, normalize the output to the maximum amplitude

        @param impulse_audio: the path to the audio or a variable of type np.ndarray that
            will be used as the convolution filter

        @param seed: the seed for the random number generator

        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
        self.impulse_audio = impulse_audio
        self.seed = seed
        self.normalize = normalize

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies a convolution to input tensor using given filter using FFT

        @param audio: the audio array to be augmented

        @param sample_rate: the audio sample rate of the inputted audio

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended

        @returns: the augmented audio array and sample rate
        """
        return F.fft_convolve(
            audio,
            sample_rate,
            self.normalize,
            self.impulse_audio,
            self.seed,
            metadata=metadata,
        )
