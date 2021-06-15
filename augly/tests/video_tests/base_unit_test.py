#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import hashlib
import os
import tempfile
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

from augly.tests import VideoAugConfig
from augly.utils import TEST_URI, pathmgr


def are_equal_videos(a_path: str, b_path: str) -> bool:
    hasher = hashlib.md5()
    with open(a_path, "rb") as afile:
        buf = afile.read()
        hasher.update(buf)
        a_md5_hash = hasher.hexdigest()

    hasher = hashlib.md5()
    with open(b_path, "rb") as bfile:
        buf = bfile.read()
        hasher.update(buf)
        b_md5_hash = hasher.hexdigest()

    return a_md5_hash == b_md5_hash


def are_equal_metadata(
    actual_meta: List[Dict[str, Any]],
    expected_meta: List[Dict[str, Any]],
    exclude_keys: Optional[List[str]],
) -> bool:
    if actual_meta == expected_meta:
        return True

    for actual_dict, expected_dict in zip(actual_meta, expected_meta):
        for (act_k, act_v), (exp_k, exp_v) in zip(
            sorted(actual_dict.items(), key=lambda kv: kv[0]),
            sorted(expected_dict.items(), key=lambda kv: kv[0]),
        ):
            if exclude_keys is not None and act_k in exclude_keys:
                continue

            if act_k != exp_k:
                return False

            if act_v == exp_v:
                continue

            """
            Allow relative paths in expected metadata: just check that the end of the
            actual path matches the expected path
            """
            condition = (
                lambda actual, expected: isinstance(actual, str)
                and isinstance(expected, str)
                and actual[-len(expected) :] == expected
            )

            if isinstance(act_v, list) and isinstance(exp_v, list):
                for actual_path, expected_path in zip(act_v, exp_v):
                    if not condition(actual_path, expected_path):
                        return False
            elif not condition(act_v, exp_v):
                return False

    return True


class BaseVideoUnitTest(unittest.TestCase):
    ref_vid_dir = os.path.join(TEST_URI, "video", "expected_output")

    def test_import(self) -> None:
        try:
            import augly.video as vidaugs
        except ImportError:
            self.fail("vidaugs failed to import")
        self.assertTrue(dir(vidaugs))

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.config, cls.local_vid_path, vid_file = cls.download_video(0)

    def evaluate_function(
        self,
        aug_function: Callable[..., None],
        ref_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        ref_filename = ref_filename or aug_function.__name__
        ref_vid_path = self.get_ref_video(ref_filename)

        if not kwargs.pop("diff_video_input", False):
            kwargs["video_path"] = self.local_vid_path

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
            aug_function(output_path=tmpfile.name, **kwargs)
            self.assertTrue(are_equal_videos(ref_vid_path, tmpfile.name))

    def evaluate_class(
        self,
        transform_class: Callable[..., None],
        fname: str,
        seed: Optional[int] = None,
        metadata_exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        metadata = []

        if not kwargs.pop("diff_video_input", False):
            kwargs["video_path"] = self.local_vid_path

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
            transform_class(
                output_path=tmpfile.name, seed=seed, metadata=metadata, **kwargs
            )
            self.assertTrue(os.path.exists(tmpfile.name))

        self.assertTrue(
            are_equal_metadata(metadata, self.metadata[fname], metadata_exclude_keys),
        )

    def get_ref_video(self, fname: str) -> str:
        ref_vid_name = f"test_{fname}.mp4"
        return pathmgr.get_local_path(os.path.join(self.ref_vid_dir, ref_vid_name))

    @staticmethod
    def download_video(input_file_index: int) -> Tuple[VideoAugConfig, str, str]:
        config = VideoAugConfig(input_file_index=input_file_index)
        vid_path, vid_file = config.get_input_path()

        local_vid_path = pathmgr.get_local_path(vid_path)
        return config, local_vid_path, vid_file
