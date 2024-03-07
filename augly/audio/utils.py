#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numbers
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from augly import utils
from augly.audio import intensity as audintensity
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
import librosa
import soundfile as sf  # @manual=fbsource//third-party/pypi/soundfile:soundfile
import torchaudio

# Use Any because np.random.Generator is not a valid type for pyre
RNG = Any
RNGSeed = Union[int, RNG]
Segment = utils.Segment


def validate_and_load_audio(
    audio: Union[str, np.ndarray], sample_rate: int = utils.DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    If audio is a str, loads the audio as an np.ndarray and returns that & the
    audio's sample rate (returned by librosa.load()). If audio is an np.ndarray,
    just returns the passed in audio & sample_rate.
    """
    if isinstance(audio, str):
        local_path = utils.pathmgr.get_local_path(audio)
        utils.validate_audio_path(local_path)
        return librosa.load(local_path, sr=None, mono=False)

    assert isinstance(
        audio, np.ndarray
    ), "Expected type np.ndarray for variable 'audio'"

    assert (
        isinstance(sample_rate, int) and sample_rate > 0
    ), "Expected 'sample_rate' to be a positive integer"

    return audio, sample_rate


def ret_and_save_audio(
    audio: np.ndarray,
    output_path: Optional[str],
    sample_rate: int = utils.DEFAULT_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    if output_path is not None:
        utils.validate_output_path(output_path)

        try:
            # Note: librosa reads in audio data as (num_channels, num_samples),
            # but soundfile expects it to be (num_samples, num_channels) when
            # writing it out, so we have to swap axes here.
            saved_audio = np.swapaxes(audio, 0, 1) if audio.ndim > 1 else audio
            sf.write(output_path, saved_audio, sample_rate)
        except TypeError:
            saved_audio = audio if audio.ndim > 1 else audio.reshape(1, audio.shape[-1])
            torchaudio.backend.sox_io_backend.save(
                output_path, torch.Tensor(saved_audio), sample_rate, channels_first=True
            )

    return audio, sample_rate


def check_random_state(seed: Optional[RNGSeed]) -> RNG:
    """
    Turn seed into a np.random.RandomState instance

    @param seed: instance of RandomState:
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a numpy.random.RandomState instance"
    )


def get_metadata(
    metadata: Optional[List[Dict[str, Any]]],
    function_name: str,
    audio: np.ndarray,
    sample_rate: int,
    dst_audio: np.ndarray,
    dst_sample_rate: int,
    **kwargs,
) -> None:
    if metadata is None:
        return

    assert isinstance(
        metadata, list
    ), "Expected 'metadata' to be set to None or of type list"

    src_duration = audio.shape[-1] / sample_rate
    dst_duration = dst_audio.shape[-1] / dst_sample_rate
    src_segments, dst_segments = compute_segments(
        function_name, src_duration, dst_duration, metadata, **kwargs
    )

    metadata.append(
        {
            "name": function_name,
            "src_duration": src_duration,
            "dst_duration": dst_duration,
            "src_num_channels": 1 if audio.ndim == 1 else audio.shape[0],
            "dst_num_channels": 1 if dst_audio.ndim == 1 else dst_audio.shape[0],
            "src_sample_rate": sample_rate,
            "dst_sample_rate": dst_sample_rate,
            "src_segments": [src_segment._asdict() for src_segment in src_segments],
            "dst_segments": [dst_segment._asdict() for dst_segment in dst_segments],
            **kwargs,
        }
    )

    intensity_kwargs = {"metadata": metadata[-1], **kwargs}
    metadata[-1]["intensity"] = getattr(
        audintensity, f"{function_name}_intensity", lambda **_: 0.0
    )(**intensity_kwargs)


def compute_changed_segments(
    name: str,
    src_segments: List[Segment],
    dst_segments: List[Segment],
    src_duration: float,
    dst_duration: float,
    speed_factor: float,
    **kwargs,
) -> Tuple[List[Segment], List[Segment]]:
    """
    This function performs the logic of computing the new matching segments based
    on the old ones, for the set of transforms that temporally change the video.

    Returns the lists of new src segments & dst segments, respectively.
    """
    new_src_segments, new_dst_segments = [], []
    for src_segment, dst_segment in zip(src_segments, dst_segments):
        if name == "insert_in_background":
            offset = kwargs["offset_factor"] * kwargs["background_duration"]
            # The matching segments are just offset in the dst audio by the amount
            # of background video inserted before the src video.
            new_src_segments.append(src_segment)
            new_dst_segments.append(dst_segment.delta(offset, offset))
        elif name == "clip":
            crop_start = kwargs["offset_factor"] * src_duration
            crop_end = crop_start + kwargs["duration_factor"] * src_duration
            utils.compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                crop_start,
                crop_end,
                new_src_segments,
                new_dst_segments,
            )
        elif name == "fft_convolve":
            new_src_segments.append(src_segment)
            new_dst_segments.append(Segment(dst_segment.start, dst_duration))
        elif name in [
            "speed",
            "tempo",
            "time_stretch",
        ]:
            # speed_factor > 1 if speedup, < 1 if slow down
            speed_factor = src_duration / dst_duration
            new_src_segments.append(src_segment)
            new_dst_segments.append(
                Segment(
                    dst_segment.start / speed_factor, dst_segment.end / speed_factor
                )
            )
    return new_src_segments, new_dst_segments


def compute_segments(
    name: str,
    src_duration: float,
    dst_duration: float,
    metadata: List[Dict[str, Any]],
    **kwargs,
) -> Tuple[List[Segment], List[Segment]]:
    speed_factor = 1.0
    if not metadata:
        src_segments = [Segment(0.0, src_duration)]
        dst_segments = [Segment(0.0, src_duration)]
    else:
        src_segments = [
            Segment(segment_dict["start"], segment_dict["end"])
            for segment_dict in metadata[-1]["src_segments"]
        ]
        dst_segments = [
            Segment(segment_dict["start"], segment_dict["end"])
            for segment_dict in metadata[-1]["dst_segments"]
        ]
        for meta in metadata:
            if meta["name"] in ["speed", "tempo"]:
                speed_factor *= meta["factor"]
            if meta["name"] == "time_stretch":
                speed_factor *= meta["rate"]

    if name in [
        "insert_in_background",
        "clip",
        "speed",
        "tempo",
        "time_stretch",
        "fft_convolve",
    ]:
        return compute_changed_segments(
            name,
            src_segments,
            dst_segments,
            src_duration,
            dst_duration,
            speed_factor,
            **kwargs,
        )
    else:
        return src_segments, dst_segments
