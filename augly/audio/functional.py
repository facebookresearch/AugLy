#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import augly.audio.utils as audutils
import numpy as np
import torch
from augly.utils import DEFAULT_SAMPLE_RATE
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
import librosa
from torchaudio import sox_effects


def add_background_noise(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    background_audio: Optional[Union[str, np.ndarray]] = None,
    snr_level_db: float = 10.0,
    seed: Optional[audutils.RNGSeed] = None,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Mixes in a background sound into the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param background_audio: the path to the background audio or a variable of type
        np.ndarray containing the background audio. If set to `None`, the background
        audio will be white noise

    @param snr_level_db: signal-to-noise ratio in dB

    @param seed: a NumPy random generator (or seed) such that the results
        remain reproducible

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(
        snr_level_db, (int, float)
    ), "Expected 'snr_level_db' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")
        func_kwargs.pop("seed")

    random_generator = audutils.check_random_state(seed)

    if background_audio is None:
        background_audio = random_generator.standard_normal(audio.shape)
    else:
        background_audio, _ = audutils.validate_and_load_audio(background_audio, 1)

    if metadata is not None:
        func_kwargs["background_duration"] = background_audio.shape[-1] / sample_rate

    audio_rms = np.sqrt(np.mean(np.square(audio), axis=-1))
    bg_rms = np.sqrt(np.mean(np.square(background_audio), axis=-1))
    desired_bg_rms = audio_rms / (10 ** (snr_level_db / 20))

    if isinstance(bg_rms, np.number) and isinstance(desired_bg_rms, np.ndarray):
        desired_bg_rms = desired_bg_rms.mean()
    elif isinstance(bg_rms, np.ndarray) and isinstance(desired_bg_rms, np.number):
        bg_rms = bg_rms.mean()
    elif isinstance(bg_rms, np.ndarray) and isinstance(desired_bg_rms, np.ndarray):
        bg_rms = bg_rms.reshape((bg_rms.shape[0], 1))
        desired_bg_rms = desired_bg_rms.reshape((desired_bg_rms.shape[0], 1))
        assert bg_rms.shape == desired_bg_rms.shape, (
            "Handling stereo audio and stereo background audio with different "
            "amounts of channels is currently unsupported"
        )

    background_audio *= desired_bg_rms / bg_rms

    while background_audio.shape[-1] < audio.shape[-1]:
        axis = 0 if background_audio.ndim == 1 else 1
        background_audio = np.concatenate(
            (background_audio, background_audio), axis=axis
        )

    background_audio = (
        background_audio[: audio.shape[-1]]
        if background_audio.ndim == 1
        else background_audio[:, : audio.shape[-1]]
    )

    aug_audio = audio + background_audio
    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="add_background_noise",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def apply_lambda(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    aug_function: Callable[..., Tuple[np.ndarray, int]] = lambda x, y: (x, y),
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    Apply a user-defined lambda to the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param aug_function: the augmentation function to be applied onto the audio (should
        expect the audio np.ndarray & sample rate int as input, and return the
        transformed audio & sample rate)

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @param **kwargs: the input attributes to be passed into `aug_function`

    @returns: the augmented audio array and sample rate
    """
    assert callable(aug_function), (
        repr(type(aug_function).__name__) + " object is not callable"
    )

    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)
    aug_audio, out_sample_rate = aug_function(audio, sample_rate, **kwargs)

    audutils.get_metadata(
        metadata=metadata,
        function_name="apply_lambda",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=out_sample_rate,
        aug_function=aug_function.__name__,
        output_path=output_path,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, out_sample_rate)


def change_volume(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume_db: float = 0.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Changes the volume of the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param volume_db: the decibel amount by which to either increase
        (positive value) or decrease (negative value) the volume of the audio

    @param output_path: the path in which the resulting audio will be stored. If
        None, the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(volume_db, (int, float)), "Expected 'volume_db' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    aug_audio = audio.reshape((num_channels, -1))

    aug_audio, out_sample_rate = sox_effects.apply_effects_tensor(
        torch.Tensor(aug_audio), sample_rate, [["vol", str(volume_db), "dB"]]
    )

    aug_audio = aug_audio.numpy()
    if num_channels == 1:
        aug_audio = aug_audio.reshape((aug_audio.shape[-1],))

    audutils.get_metadata(
        metadata=metadata,
        function_name="change_volume",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=sample_rate,
        volume_db=volume_db,
        output_path=output_path,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def clicks(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seconds_between_clicks: float = 0.5,
    snr_level_db: float = 1.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Adds clicks to the audio at a given regular interval

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param seconds_between_clicks: the amount of time between each click that
        will be added to the audio, in seconds

    @param snr_level_db: signal-to-noise ratio in dB

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(
        seconds_between_clicks, (int, float)
    ), "Expected 'seconds_between_clicks' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    num_samples = audio.shape[-1]
    seconds_in_audio = num_samples / sample_rate
    times = np.arange(0, seconds_in_audio, seconds_between_clicks)
    clicks_audio = librosa.clicks(times=times, sr=sample_rate)

    aug_audio, out_sample_rate = add_background_noise(
        audio,
        sample_rate=sample_rate,
        background_audio=clicks_audio,
        snr_level_db=snr_level_db,
    )

    audutils.get_metadata(
        metadata=metadata,
        function_name="clicks",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=out_sample_rate,
        seconds_between_clicks=seconds_between_clicks,
        output_path=output_path,
        clicks_duration=clicks_audio.shape[-1] / sample_rate,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, out_sample_rate)


def clip(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    offset_factor: float = 0.0,
    duration_factor: float = 1.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Clips the audio using the specified offset and duration factors

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param offset_factor: start point of the crop relative to the audio duration
        (this parameter is multiplied by the audio duration)

    @param duration_factor: the length of the crop relative to the audio duration
        (this parameter is multiplied by the audio duration)

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        0.0 <= (offset_factor + duration_factor) <= 1.0
    ), "Combination of offset and duration factors exceed audio length"

    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    num_samples = audio.shape[-1]
    start = int(offset_factor * num_samples)
    end = int((offset_factor + duration_factor) * num_samples)
    aug_audio = audio[..., start:end]

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="clip",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            start_sample=start,
            end_sample=end,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def harmonic(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Extracts the harmonic part of the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param kernel_size: kernel size for the median filters

    @param power: exponent for the Wiener filter when constructing soft
        mask matrices

    @param margin: margin size for the masks

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(kernel_size, int), "Expected 'kernel_size' to be an int"
    assert isinstance(power, (int, float)), "Expected 'power' to be a number"
    assert isinstance(margin, (int, float)), "Expected 'margin' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    if num_channels == 1:
        aug_audio = librosa.effects.harmonic(
            audio, kernel_size=kernel_size, power=power, margin=margin
        )
    else:
        aug_audio = np.vstack(
            [
                librosa.effects.harmonic(
                    np.asfortranarray(audio[c]),
                    kernel_size=kernel_size,
                    power=power,
                    margin=margin,
                )
                for c in range(num_channels)
            ]
        )

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="harmonic",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def high_pass_filter(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    cutoff_hz: float = 3000.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Allows audio signals with a frequency higher than the given cutoff to pass
    through and attenuates signals with frequencies lower than the cutoff frequency

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param cutoff_hz: frequency (in Hz) where signals with lower frequencies will
        begin to be reduced by 6dB per octave (doubling in frequency) below this point

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(cutoff_hz, (int, float)), "Expected 'cutoff_hz' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    rc = 1 / (2 * math.pi * cutoff_hz)
    dt = 1 / sample_rate
    alpha = rc / (rc + dt)
    num_channels = 1 if audio.ndim == 1 else audio.shape[0]

    if num_channels == 1:
        audio = audio.reshape(1, audio.shape[0])

    frame_count = audio.shape[1]
    high_pass_array = np.zeros(audio.shape)

    for i in range(num_channels):
        high_pass_array[i][0] = audio[i][0]
        for j in range(1, frame_count):
            high_pass_array[i][j] = alpha * (
                high_pass_array[i][j - 1] + audio[i][j] - audio[i][j - 1]
            )

    if num_channels == 1:
        high_pass_array = high_pass_array.reshape((high_pass_array.shape[1],))

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="high_pass_filter",
            dst_audio=high_pass_array,
            dst_sample_rate=sample_rate,
            alpha=alpha,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(high_pass_array, output_path, sample_rate)


def insert_in_background(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    offset_factor: float = 0.0,
    background_audio: Optional[Union[str, np.ndarray]] = None,
    seed: Optional[audutils.RNGSeed] = None,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Inserts audio into a background clip in a non-overlapping manner.

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param offset_factor: insert point relative to the background duration
        (this parameter is multiplied by the background duration)

    @param background_audio: the path to the background audio or a variable of type
        np.ndarray containing the background audio. If set to `None`, the background
        audio will be white noise, with the same duration as the audio.

    @param seed: a NumPy random generator (or seed) such that the results
        remain reproducible

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        0.0 <= offset_factor <= 1.0
    ), "Expected 'offset_factor' to be a number in the range [0, 1]"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")
        func_kwargs.pop("seed")

    random_generator = audutils.check_random_state(seed)
    if background_audio is None:
        background_audio = random_generator.standard_normal(audio.shape)
    else:
        background_audio, _ = audutils.validate_and_load_audio(
            background_audio, sample_rate
        )
        num_channels = 1 if audio.ndim == 1 else audio.shape[0]
        bg_num_channels = 1 if background_audio.ndim == 1 else background_audio.shape[0]
        if bg_num_channels != num_channels:
            background_audio, _background_sr = to_mono(background_audio)
            if num_channels > 1:
                background_audio = np.tile(background_audio, (num_channels, 1))

    num_samples_bg = background_audio.shape[-1]
    offset = int(offset_factor * num_samples_bg)
    aug_audio = np.hstack(
        [background_audio[..., :offset], audio, background_audio[..., offset:]]
    )

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="insert_in_background",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            background_duration=background_audio.shape[-1] / sample_rate,
            offset=offset,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def invert_channels(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Inverts channels of the audio.
    If the audio has only one channel, no change is applied.
    Otherwise, it inverts the order of the channels, eg for 4 channels,
    it returns channels in order [3, 2, 1, 0].

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    aug_audio = audio
    if audio.ndim > 1:
        num_channels = audio.shape[0]
        inverted_channels = list(range(num_channels))[::-1]
        aug_audio = audio[inverted_channels, :]

    audutils.get_metadata(
        metadata=metadata,
        function_name="invert_channels",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=sample_rate,
        output_path=output_path,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def low_pass_filter(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    cutoff_hz: float = 500.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Allows audio signals with a frequency lower than the given cutoff to pass through
    and attenuates signals with frequencies higher than the cutoff frequency

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param cutoff_hz: frequency (in Hz) where signals with higher frequencies will
        begin to be reduced by 6dB per octave (doubling in frequency) above this point

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(cutoff_hz, (int, float)), "Expected 'cutoff_hz' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    rc = 1 / (2 * math.pi * cutoff_hz)
    dt = 1 / sample_rate
    alpha = dt / (rc + dt)
    num_channels = 1 if audio.ndim == 1 else audio.shape[0]

    if num_channels == 1:
        audio = audio.reshape(1, audio.shape[0])

    frame_count = audio.shape[1]
    low_pass_array = np.zeros(audio.shape)

    for i in range(num_channels):
        low_pass_array[i][0] = alpha * audio[i][0]
        for j in range(1, frame_count):
            low_pass_array[i][j] = low_pass_array[i][j - 1] + alpha * (
                audio[i][j] - low_pass_array[i][j - 1]
            )

    if num_channels == 1:
        low_pass_array = low_pass_array.reshape((low_pass_array.shape[1],))

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="low_pass_filter",
            dst_audio=low_pass_array,
            dst_sample_rate=sample_rate,
            alpha=alpha,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(low_pass_array, output_path, sample_rate)


def normalize(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    norm: Optional[float] = np.inf,
    axis: int = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Normalizes the audio array along the chosen axis (norm(audio, axis=axis) == 1)

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param norm: the type of norm to compute:
        - np.inf: maximum absolute value
        - -np.inf: minimum absolute value
        - 0: number of non-zeros (the support)
        - float: corresponding l_p norm
        - None: no normalization is performed

    @param axis: axis along which to compute the norm

    @param threshold: if provided, only the columns (or rows) with norm of at
        least `threshold` are normalized

    @param fill: if None, then columns (or rows) with norm below `threshold` are left
        as is. If False, then columns (rows) with norm below `threshold` are set to 0.
        If True, then columns (rows) with norm below `threshold` are filled uniformly
        such that the corresponding norm is 1

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        isinstance(axis, int) and axis >= 0
    ), "Expected 'axis' to be a nonnegative number"
    assert threshold is None or isinstance(
        threshold, (int, float)
    ), "Expected 'threshold' to be a number or None"
    assert fill is None or isinstance(
        fill, bool
    ), "Expected 'threshold' to be a boolean or None"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs["norm"] = str(func_kwargs["norm"])
        func_kwargs.pop("metadata")

    aug_audio = librosa.util.normalize(audio, norm, axis, threshold, fill)

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="normalize",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def peaking_equalizer(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    center_hz: float = 500.0,
    q: float = 1.0,
    gain_db: float = -3.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Applies a two-pole peaking equalization filter. The signal-level at and around
    `center_hz` can be increased or decreased, while all other frequencies are unchanged

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param center_hz: point in the frequency spectrum at which EQ is applied

    @param q: ratio of center frequency to bandwidth; bandwidth is inversely
        proportional to Q, meaning that as you raise Q, you narrow the bandwidth

    @param gain_db: amount of gain (boost) or reduction (cut) that is applied at a
        given frequency. Beware of clipping when using positive gain

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(center_hz, (int, float)), "Expected 'center_hz' to be a number"
    assert isinstance(q, (int, float)) and q > 0, "Expected 'q' to be a positive number"
    assert isinstance(gain_db, (int, float)), "Expected 'gain_db' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    aug_audio = audio.reshape((num_channels, -1))

    aug_audio, out_sample_rate = sox_effects.apply_effects_tensor(
        torch.Tensor(aug_audio),
        sample_rate,
        [["equalizer", str(center_hz), f"{q}q", str(gain_db)]],
    )

    aug_audio = aug_audio.numpy()
    if num_channels == 1:
        aug_audio = aug_audio.reshape((aug_audio.shape[-1],))

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="peaking_equalizer",
            dst_audio=aug_audio,
            dst_sample_rate=out_sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, out_sample_rate)


def percussive(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Extracts the percussive part of the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param kernel_size: kernel size for the median filters

    @param power: exponent for the Wiener filter when constructing soft mask matrices

    @param margin: margin size for the masks

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(kernel_size, int), "Expected 'kernel_size' to be an int"
    assert isinstance(power, (int, float)), "Expected 'power' to be a number"
    assert isinstance(margin, (int, float)), "Expected 'margin' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    if num_channels == 1:
        aug_audio = librosa.effects.percussive(
            audio, kernel_size=kernel_size, power=power, margin=margin
        )
    else:
        aug_audio = np.vstack(
            [
                librosa.effects.percussive(
                    np.asfortranarray(audio[c]),
                    kernel_size=kernel_size,
                    power=power,
                    margin=margin,
                )
                for c in range(num_channels)
            ]
        )

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="percussive",
            dst_audio=aug_audio,
            dst_sample_rate=sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def pitch_shift(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_steps: float = 1.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Shifts the pitch of the audio by `n_steps`

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param n_steps: each step is equal to one semitone

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(n_steps, (int, float)), "Expected 'n_steps' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)
    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    if num_channels == 1:
        aug_audio = librosa.effects.pitch_shift(audio, sample_rate, n_steps)
    else:
        aug_audio = np.vstack(
            [
                librosa.effects.pitch_shift(
                    np.asfortranarray(audio[c]), sample_rate, n_steps
                )
                for c in range(num_channels)
            ]
        )

    audutils.get_metadata(
        metadata=metadata,
        function_name="pitch_shift",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=sample_rate,
        output_path=output_path,
        n_steps=n_steps,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def reverb(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    reverberance: float = 50.0,
    hf_damping: float = 50.0,
    room_scale: float = 100.0,
    stereo_depth: float = 100.0,
    pre_delay: float = 0.0,
    wet_gain: float = 0.0,
    wet_only: bool = False,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Adds reverberation to the audio

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param reverberance: (%) sets the length of the reverberation tail. This determines
        how long the reverberation continues for after the original sound being reverbed
        comes to an end, and so simulates the "liveliness" of the room acoustics

    @param hf_damping: (%) increasing the damping produces a more "muted" effect. The
        reverberation does not build up as much, and the high frequencies decay faster
        than the low frequencies

    @param room_scale: (%) sets the size of the simulated room. A high value will
        simulate the reverberation effect of a large room and a low value will simulate
        the effect of a small room

    @param stereo_depth: (%) sets the apparent "width" of the reverb effect for stereo
        tracks only. Increasing this value applies more variation between left and right
        channels, creating a more "spacious" effect. When set at zero, the effect is
        applied independently to left and right channels

    @param pre_delay: (ms) delays the onset of the reverberation for the set time after
        the start of the original input. This also delays the onset of the reverb tail

    @param wet_gain: (db) applies volume adjustment to the reverberation ("wet")
        component in the mix

    @param wet_only: only the wet signal (added reverberation) will be in the resulting
        output, and the original audio will be removed

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert isinstance(
        reverberance, (int, float)
    ), "Expected 'reverberance' to be a number"
    assert isinstance(hf_damping, (int, float)), "Expected 'hf_damping' to be a number"
    assert isinstance(room_scale, (int, float)), "Expected 'room_scale' to be a number"
    assert isinstance(
        stereo_depth, (int, float)
    ), "Expected 'stereo_depth' to be a number"
    assert isinstance(pre_delay, (int, float)), "Expected 'pre_delay' to be a number"
    assert isinstance(wet_gain, (int, float)), "Expected 'wet_gain' to be a number"
    assert isinstance(wet_only, bool), "Expected 'wet_only' to be a number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    if metadata is not None:
        func_kwargs = deepcopy(locals())
        func_kwargs.pop("metadata")

    aug_audio = audio.reshape((1, audio.shape[-1])) if audio.ndim == 1 else audio

    effect = ["reverb"]
    if wet_only:
        effect.append("-w")

    aug_audio, out_sample_rate = sox_effects.apply_effects_tensor(
        torch.Tensor(aug_audio),
        sample_rate,
        [
            effect
            + [
                str(reverberance),
                str(hf_damping),
                str(room_scale),
                str(stereo_depth),
                str(pre_delay),
                str(wet_gain),
            ]
        ],
    )

    aug_audio = aug_audio.numpy()
    if audio.shape[0] == 1:
        aug_audio = aug_audio.reshape((aug_audio.shape[-1],))

    if metadata is not None:
        audutils.get_metadata(
            metadata=metadata,
            function_name="reverb",
            dst_audio=aug_audio,
            dst_sample_rate=out_sample_rate,
            # pyre-fixme[61]: `func_kwargs` may not be initialized here.
            **func_kwargs,
        )

    return audutils.ret_and_save_audio(aug_audio, output_path, out_sample_rate)


def speed(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    factor: float = 2.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Changes the speed of the audio, affecting pitch as well

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param factor: the speed factor. If rate > 1 the audio will be sped up by that
        factor; if rate < 1 the audio will be slowed down by that factor

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        isinstance(factor, (int, float)) and factor > 0
    ), "Expected 'factor' to be a positive number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    out_sample_rate = int(sample_rate * factor)
    audutils.get_metadata(
        metadata=metadata,
        function_name="speed",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=audio,
        dst_sample_rate=out_sample_rate,
        output_path=output_path,
        factor=factor,
    )

    return audutils.ret_and_save_audio(audio, output_path, out_sample_rate)


def tempo(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    factor: float = 2.0,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Adjusts the tempo of the audio by a given factor

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param factor: the tempo factor. If rate > 1 the audio will be sped up by that
        factor; if rate < 1 the audio will be slowed down by that factor, without
        affecting the pitch

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        isinstance(factor, (int, float)) and factor > 0
    ), "Expected 'factor' to be a positive number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)

    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    aug_audio = audio.reshape((num_channels, -1))

    aug_audio, out_sample_rate = sox_effects.apply_effects_tensor(
        torch.Tensor(aug_audio), sample_rate, [["tempo", str(factor)]]
    )

    aug_audio = aug_audio.numpy()
    if num_channels == 1:
        aug_audio = aug_audio.reshape((aug_audio.shape[-1],))

    audutils.get_metadata(
        metadata=metadata,
        function_name="tempo",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=out_sample_rate,
        output_path=output_path,
        factor=factor,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, out_sample_rate)


def time_stretch(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    rate: float = 1.5,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Time-stretches the audio by a fixed rate

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param rate: the time stretch factor. If rate > 1 the audio will be sped up by
        that factor; if rate < 1 the audio will be slowed down by that factor

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    assert (
        isinstance(rate, (int, float)) and rate > 0
    ), "Expected 'rate' to be a positive number"
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)
    num_channels = 1 if audio.ndim == 1 else audio.shape[0]
    if num_channels == 1:
        aug_audio = librosa.effects.time_stretch(audio, rate)
    else:
        aug_audio = np.vstack(
            [
                librosa.effects.time_stretch(np.asfortranarray(audio[c]), rate)
                for c in range(num_channels)
            ]
        )

    audutils.get_metadata(
        metadata=metadata,
        function_name="time_stretch",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=sample_rate,
        output_path=output_path,
        rate=rate,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)


def to_mono(
    audio: Union[str, np.ndarray],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Converts the audio from stereo to mono by averaging samples across channels

    @param audio: the path to the audio or a variable of type np.ndarray that
        will be augmented

    @param sample_rate: the audio sample rate of the inputted audio

    @param output_path: the path in which the resulting audio will be stored. If None,
        the resulting np.ndarray will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest duration, sample rates, etc. will be
        appended to the inputted list. If set to None, no metadata will be appended

    @returns: the augmented audio array and sample rate
    """
    audio, sample_rate = audutils.validate_and_load_audio(audio, sample_rate)
    aug_audio = librosa.core.to_mono(audio)

    audutils.get_metadata(
        metadata=metadata,
        function_name="to_mono",
        audio=audio,
        sample_rate=sample_rate,
        dst_audio=aug_audio,
        dst_sample_rate=sample_rate,
        output_path=output_path,
    )

    return audutils.ret_and_save_audio(aug_audio, output_path, sample_rate)
