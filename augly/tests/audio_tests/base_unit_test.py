#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from typing import Callable

import numpy as np
from augly.tests import AudioAugConfig
from augly.utils import pathmgr, TEST_URI
from augly.utils.libsndfile import install_libsndfile


install_libsndfile()
import librosa  # fmt: off
import soundfile as sf  # fmt: off


def are_equal_audios(a: np.ndarray, b: np.ndarray) -> bool:
    return a.size == b.size and np.allclose(a, b, atol=1e-4)


class BaseAudioUnitTest(unittest.TestCase):
    ref_audio_dir = os.path.join(TEST_URI, "audio", "speech_commands_expected_output")
    local_audio_paths = []
    audios = []
    sample_rates = []
    metadata = {}

    def test_import(self) -> None:
        try:
            from augly import audio as audaugs
        except ImportError:
            self.fail("audaugs failed to import")
        self.assertTrue(dir(audaugs), "Audio directory does not exist")

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        config = AudioAugConfig()

        for i in range(len(config.input_files)):
            audio_path, audio_file = config.get_input_path(i)
            local_audio_path = pathmgr.get_local_path(audio_path)
            audio, sample_rate = librosa.load(local_audio_path, sr=None, mono=False)

            cls.audios.append(audio)
            cls.sample_rates.append(sample_rate)
            cls.local_audio_paths.append(local_audio_path)

    def evaluate_function(self, aug_function: Callable[..., np.ndarray], **kwargs):
        folders = ["mono", "stereo"]

        for i, local_audio_path in enumerate(self.local_audio_paths):
            ref = self.get_ref_audio(aug_function.__name__, folders[i])

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                np.random.seed(1)
                aug_function(local_audio_path, output_path=tmpfile.name, **kwargs)
                dst = librosa.load(tmpfile.name, sr=None, mono=False)[0]

            self.assertTrue(
                are_equal_audios(dst, ref), "Expected and outputted audio do not match"
            )

    def evaluate_class(self, transform_class: Callable[..., np.ndarray], fname: str):
        metadata = []
        audio, sample_rate = transform_class(
            self.audios[0], self.sample_rates[0], metadata
        )

        self.assertEqual(
            metadata,
            self.metadata[fname],
            "Expected and outputted metadata do not match",
        )

        if audio.ndim > 1:
            audio = np.swapaxes(audio, 0, 1)

        # we compare the audio arrays loaded in from the audio files rather than the
        # returned audio array and the loaded in audio array because the array is
        # slightly modified during sf.write
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio, sample_rate)
            dst = librosa.load(tmpfile.name, sr=None, mono=False)[0]

        ref = self.get_ref_audio(fname)
        self.assertTrue(
            are_equal_audios(dst, ref), "Expected and outputted audio do not match"
        )

    def get_ref_audio(self, fname: str, folder: str = "mono") -> np.ndarray:
        local_ref_path = pathmgr.get_local_path(
            os.path.join(self.ref_audio_dir, folder, f"test_{fname}.wav")
        )

        return librosa.load(local_ref_path, sr=None, mono=False)[0]

    @classmethod
    def tearDownClass(cls):
        cls.local_audio_paths = []
        cls.audios = []
        cls.sample_rates = []
