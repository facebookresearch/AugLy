#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from augly.utils import Segment
from augly.utils.base_paths import ASSETS_BASE_DIR
from augly.video import helpers

from augly.video.augmenters import ffmpeg as af


class MetadataUnitTest(unittest.TestCase):
    def assert_equal_segments(self, actual: Segment, expected: Segment):
        fail_msg = f"actual={actual}, expected={expected}"
        self.assertAlmostEqual(actual.start, expected.start, msg=fail_msg)
        self.assertAlmostEqual(actual.end, expected.end, msg=fail_msg)
        self.assertEqual(actual.src_id, expected.src_id, msg=fail_msg)

    def assert_equal_segment_lists(
        self, actual: List[Segment], expected: List[Segment]
    ):
        self.assertEqual(
            len(actual), len(expected), f"actuals={actual}, expected={expected}"
        )
        for act, exp in zip(actual, expected):
            self.assert_equal_segments(act, exp)

    def test_insert_in_background_transitions_start(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "insert_in_background",
            src_duration=20.0,
            dst_duration=26.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            offset_factor=0.0,
            background_video_duration=10.0,
            source_percentage=None,
            transition_before=False,
            transition_after=True,
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=4.0
            ),
        )
        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=0.0, end=18.0)]
        )
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=0.0, end=18.0)]
        )

    def test_insert_in_background_transitions_end(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "insert_in_background",
            src_duration=20.0,
            dst_duration=26.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            offset_factor=1.0,
            background_video_duration=10.0,
            source_percentage=None,
            transition_before=True,
            transition_after=False,
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=4.0
            ),
        )
        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=2.0, end=20.0)]
        )
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=8.0, end=26.0)]
        )

    def test_insert_in_background_transitions(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "insert_in_background",
            src_duration=20.0,
            dst_duration=22.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            offset_factor=0.5,
            background_video_duration=10.0,
            source_percentage=None,
            transition_before=True,
            transition_after=True,
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=4.0
            ),
        )
        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=2.0, end=18.0)]
        )
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=3.0, end=19.0)]
        )

    def test_insert_in_background(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "insert_in_background",
            src_duration=20.0,
            dst_duration=30.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            offset_factor=0.5,
            background_video_duration=10.0,
            source_percentage=None,
            transition_before=True,
            transition_after=True,
            transition=None,
        )
        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=0.0, end=20.0)]
        )
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=5.0, end=25.0)]
        )

    def test_compute_insert_in_background_multiple_segments(self):
        src_1, dst_1 = helpers.compute_segments(
            name="insert_in_background_multiple",
            src_duration=20.0,
            dst_duration=40.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            src_segment_starts=[2.0, 3.0],
            src_segment_ends=[12.0, 13.0],
            bkg_insertion_points=[5.0, 10.0],
            src_ids=["0", "1"],
            transition=None,
        )
        self.assert_equal_segment_lists(
            src_1, [Segment(2.0, 12.0, "0"), Segment(3.0, 13.0, "1")]
        )
        self.assert_equal_segment_lists(
            dst_1, [Segment(5.0, 15.0), Segment(20.0, 30.0)]
        )

        src_2, dst_2 = helpers.compute_segments(
            name="insert_in_background_multiple",
            src_duration=20.0,
            dst_duration=50.0,
            src_fps=25,
            dst_fps=30,
            metadata=None,
            src_segment_starts=[2.0, 3.0, 7.0],
            src_segment_ends=[12.0, 13.0, 20.0],
            bkg_insertion_points=[5.0, 10.0, 20.0],
            src_ids=["0", "1", "2"],
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=2.0
            ),
        )
        self.assert_equal_segment_lists(
            src_2,
            [Segment(3.0, 11.0, "0"), Segment(4.0, 12.0, "1"), Segment(8.0, 19.0, "2")],
        )
        self.assert_equal_segment_lists(
            dst_2, [Segment(4.0, 12.0), Segment(15.0, 23.0), Segment(31.0, 42.0)]
        )

    def test_time_decimate(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "time_decimate",
            src_duration=20,
            dst_duration=12,
            src_fps=25.0,
            dst_fps=30.0,
            metadata=None,
            start_offset_factor=0.1,  # 2sec
            on_factor=0.2,  # 4sec
            off_factor=0.5,  # 2sec (relative to on_factor)
            transition=None,
        )
        self.assert_equal_segment_lists(
            new_src_segments,
            [
                Segment(start=2.0, end=6.0),
                Segment(start=8.0, end=12.0),
                Segment(start=14.0, end=18.0),
            ],
        )
        self.assert_equal_segment_lists(
            new_dst_segments,
            [
                Segment(start=0, end=4.0),
                Segment(start=4.0, end=8.0),
                Segment(start=8.0, end=12.0),
            ],
        )

    def test_time_decimate_with_transition(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "time_decimate",
            src_duration=20,
            dst_duration=8,
            src_fps=25.0,
            dst_fps=30.0,
            metadata=None,
            start_offset_factor=0.1,  # 2sec
            on_factor=0.2,  # 4sec
            off_factor=0.5,  # 2sec (relative to on_factor)
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=2.0
            ),
        )
        self.assert_equal_segment_lists(
            new_src_segments,
            [
                Segment(start=2.0, end=5.0),
                Segment(start=9.0, end=11.0),
                Segment(start=15.0, end=18.0),
            ],
        )
        self.assert_equal_segment_lists(
            new_dst_segments,
            [
                Segment(start=0, end=3.0),
                Segment(start=3.0, end=5.0),
                Segment(start=5.0, end=8.0),
            ],
        )

    def test_change_video_speed(self):
        md = {
            "name": "overlay_text",
            "src_duration": 58.591925,
            "dst_duration": 58.591925,
            "src_fps": 29.952932801822325,
            "dst_fps": 29.95293280069612,
            "src_width": 640,
            "src_height": 352,
            "dst_width": 640,
            "dst_height": 352,
            "src_segments": [
                {"start": 0.0, "end": 14.492321571512164},
                {"start": 23.479556369432764, "end": 37.97187794094493},
            ],
            "dst_segments": [
                {"start": 8.791720999383276, "end": 23.284042570895444},
                {"start": 23.284042570895444, "end": 37.7763641424076},
            ],
            "text_len": 12,
            "text_change_nth": None,
            "fonts": [
                (
                    os.path.join(
                        ASSETS_BASE_DIR,
                        "similarity/media_assets/fonts/cac_champagne.ttf",
                    ),
                    os.path.join(
                        ASSETS_BASE_DIR,
                        "similarity/media_assets/fonts/cac_champagne.pkl",
                    ),
                )
            ],
            "fontscales": [0.2973747861954241, 0.5916911269561087],
            "colors": [(238, 166, 244)],
            "thickness": None,
            "random_movement": False,
            "topleft": [0.09272467655824976, 0.5098042791327592],
            "bottomright": [0.8126272414865852, 0.7849615824118924],
            "kwargs": {},
            "intensity": 0.1980864483894119,
        }

        new_src_segments, new_dst_segments = helpers.compute_segments(
            "change_video_speed",
            src_duration=58.591925,
            dst_duration=40.563641,
            src_fps=29.9,
            dst_fps=29.9,
            metadata=[
                md,
            ],
            factor=1.44,
        )

        self.assert_equal_segment_lists(
            new_src_segments,
            [
                Segment(start=0.0, end=14.492321571512164),
                Segment(start=23.479556369432764, end=37.97187794094493),
            ],
        )

        self.assert_equal_segment_lists(
            new_dst_segments,
            [
                Segment(start=6.105361805127275, end=16.169474007566283),
                Segment(start=16.169474007566283, end=26.233586210005278),
            ],
        )

    @patch(
        "augly.video.helpers.get_video_info",
        side_effect=[
            {"duration": "20.0"},
        ],
    )
    def test_concat_transition_video_in_the_middle(
        self, mock_get_video_info: MagicMock
    ):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "concat",
            src_duration=30.0,
            dst_duration=52.0,
            src_fps=29.97,
            dst_fps=30.0,
            metadata=None,
            video_paths=["before", "main", "after"],
            src_video_path_index=1,
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=4.0
            ),
        )

        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=2.0, end=28.0)]
        )
        assert mock_get_video_info.call_count == 1, mock_get_video_info.call_count
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=18.0, end=44.0)]
        )

    def test_concat_transition_main_video_first(self):
        new_src_segments, new_dst_segments = helpers.compute_segments(
            "concat",
            src_duration=30.33,
            dst_duration=64.8,
            src_fps=29.97,
            dst_fps=30.0,
            metadata=None,
            video_paths=["main", "after"],
            src_video_path_index=0,
            transition=af.TransitionConfig(
                effect=af.TransitionEffect.CIRCLECLOSE, duration=4.0
            ),
        )

        self.assert_equal_segment_lists(
            new_src_segments, [Segment(start=0.0, end=28.33)]
        )
        self.assert_equal_segment_lists(
            new_dst_segments, [Segment(start=0.0, end=28.33)]
        )
