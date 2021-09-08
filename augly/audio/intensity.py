#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


"""
This file contains 'intensity' functions for each of our augmentations.
Intensity functions give a float representation of how intense a particular
transform called with particular parameters is.

Each intensity function expects as parameters the kwargs the corresponding
transform was called with, as well as the metadata dictionary computed by the
transform (e.g. metadata[-1] after passing a list 'metadata' into the transform).

All intensity functions are normalized to be in [0, 100]. This means we are
assuming a range of valid values for each param - e.g. for pitch_shift we
assume we will never pitch shift more than an octave, meaning the range of
valid values for n_steps is [-12.0, 12.0], which you can see is assumed in the
intensity function below.
"""


def add_background_noise_intensity(snr_level_db: float = 10.0, **kwargs) -> float:
    assert isinstance(snr_level_db, (float, int)), "snr_level_db must be a number"

    max_snr_level_db_val = 110.0
    return min(
        ((max_snr_level_db_val - snr_level_db) / max_snr_level_db_val) * 100.0, 100.0
    )


def apply_lambda_intensity(
    aug_function: Callable[..., Tuple[np.ndarray, int]],
    **kwargs,
) -> float:
    intensity_func = globals().get(f"{aug_function}_intensity")
    return intensity_func(**kwargs) if intensity_func else 100.0


def change_volume_intensity(volume_db: float = 0.0, **kwargs) -> float:
    assert isinstance(volume_db, (float, int)), "volume_db must be a nonnegative number"

    max_volume_db_val = 110.0
    return min((abs(volume_db) / max_volume_db_val) * 100.0, 100.0)


def clicks_intensity(
    seconds_between_clicks: float = 0.5, snr_level_db: float = 1.0, **kwargs
) -> float:
    assert (
        isinstance(seconds_between_clicks, (float, int)) and seconds_between_clicks >= 0
    ), "seconds_between_clicks must be a nonnegative number"
    assert isinstance(snr_level_db, (float, int)), "snr_level_db must be a number"
    max_seconds_between_clicks_val = 60.0
    max_snr_level_db_val = 110.0
    seconds_between_clicks_intensity = (
        max_seconds_between_clicks_val - seconds_between_clicks
    ) / max_seconds_between_clicks_val
    snr_level_db_intensity = (
        max_snr_level_db_val - snr_level_db
    ) / max_snr_level_db_val
    return min(
        (seconds_between_clicks_intensity * snr_level_db_intensity) * 100.0, 100.0
    )


def clip_intensity(duration_factor: float = 1.0, **kwargs) -> float:
    assert 0 < duration_factor <= 1, "duration_factor must be a number in (0, 1]"

    max_duration_factor = 1.0
    return min(
        ((max_duration_factor - duration_factor) / max_duration_factor) * 100.0, 100.0
    )


def harmonic_intensity(**kwargs) -> float:
    return 100.0


def high_pass_filter_intensity(cutoff_hz: float = 3000.0, **kwargs) -> float:
    assert (
        isinstance(cutoff_hz, (float, int)) and cutoff_hz >= 0
    ), "cutoff_hz must be a nonnegative number"

    max_cutoff_hz_val = 20000.0
    return min((cutoff_hz / max_cutoff_hz_val) * 100.0, 100.0)


def insert_in_background_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    bg_to_src_duration_ratio = (
        metadata["dst_duration"] - metadata["src_duration"]
    ) / metadata["dst_duration"]
    return bg_to_src_duration_ratio * 100.0


def invert_channels_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return 0.0 if metadata["src_num_channels"] == 1 else 100.0


def low_pass_filter_intensity(cutoff_hz: float = 500.0, **kwargs) -> float:
    assert (
        isinstance(cutoff_hz, (float, int)) and cutoff_hz >= 0
    ), "cutoff_hz must be a nonnegative number"

    max_cutoff_hz_val = 20000.0
    return min(((max_cutoff_hz_val - cutoff_hz) / max_cutoff_hz_val) * 100.0, 100.0)


def normalize_intensity(norm: Optional[float] = np.inf, **kwargs) -> float:
    return 100.0 if norm else 0.0


def peaking_equalizer_intensity(q: float, gain_db: float, **kwargs) -> float:
    assert isinstance(q, (int, float)) and q > 0, "Expected 'q' to be a positive number"
    assert isinstance(gain_db, (int, float)), "Expected 'gain_db' to be a number"

    max_q_val, max_gain_db_val = 46, 110.0
    q_intensity = (max_q_val - q) / max_q_val
    gain_db_intensity = abs(gain_db) / max_gain_db_val

    return min((q_intensity * gain_db_intensity) * 100.0, 100.0)


def percussive_intensity(**kwargs) -> float:
    return 100.0


def pitch_shift_intensity(n_steps: float = 2.0, **kwargs) -> float:
    assert isinstance(n_steps, (float, int)), "n_steps must be a number"
    max_nsteps_val = 84.0
    return min((abs(n_steps) / max_nsteps_val) * 100.0, 100.0)


def reverb_intensity(
    reverberance: float = 50.0,
    wet_only: bool = False,
    room_scale: float = 100.0,
    **kwargs,
) -> float:
    assert (
        isinstance(reverberance, (float, int))
        and 0 <= reverberance <= 100
        and isinstance(room_scale, (float, int))
        and 0 <= room_scale <= 100
    ), "reverberance & room_scale must be numbers in [0, 100]"

    if wet_only:
        return 100.0
    max_reverberance_val = 100.0
    max_room_scale_val = 100.0
    return min(
        (reverberance / max_reverberance_val)
        * (room_scale / max_room_scale_val)
        * 100.0,
        100.0,
    )


def speed_intensity(factor: float = 2.0, **kwargs) -> float:
    assert (
        isinstance(factor, (float, int)) and factor > 0
    ), "factor must be a positive number"

    if factor == 1.0:
        return 0.0
    max_factor_val = 10.0
    # We want the intensity of factor = 2 to be the same as the intensity of
    # factor = 0.5, since they both change the speed by 2x.
    # speed_change_factor represents how much the speed of the audio has changed,
    # with a value in [1, inf).
    speed_change_factor = factor if factor >= 1 else 1 / factor
    return min((speed_change_factor / max_factor_val) * 100.0, 100.0)


def tempo_intensity(factor: float = 2.0, **kwargs) -> float:
    assert (
        isinstance(factor, (float, int)) and factor > 0
    ), "factor must be a positive number"

    if factor == 1.0:
        return 0.0
    max_factor_val = 10.0
    speed_change_factor = factor if factor >= 1 else 1 / factor
    return min((speed_change_factor / max_factor_val) * 100.0, 100.0)


def time_stretch_intensity(rate: float = 1.5, **kwargs) -> float:
    assert (
        isinstance(rate, (float, int)) and rate > 0
    ), "factor must be a positive number"

    if rate == 1.0:
        return 0.0
    max_rate_val = 10.0
    speed_change_rate = rate if rate >= 1 else 1 / rate
    return min((speed_change_rate / max_rate_val) * 100.0, 100.0)


def to_mono_intensity(metadata: Dict[str, Any], **kwargs) -> float:
    return 0.0 if metadata["src_num_channels"] == 1 else 100.0
