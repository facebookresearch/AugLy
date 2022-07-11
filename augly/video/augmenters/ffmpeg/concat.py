#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import List, Optional

from augly.utils import pathmgr
from augly.video.augmenters.ffmpeg.base_augmenter import BaseVidgearFFMPEGAugmenter
from augly.video.helpers import get_video_info
from dataclasses_json import dataclass_json


log = logging.getLogger(__name__)


class TransitionEffect(Enum):
    DISSOLVE = 2
    RADIAL = 3
    CIRCLEOPEN = 4
    CIRCLECLOSE = 5
    PIXELIZE = 6
    HLSLICE = 7
    HRSLICE = 8
    VUSLICE = 9
    VDSLICE = 10
    HBLUR = 11
    FADEGRAYS = 12
    FADEBLACK = 13
    FADEWHITE = 14
    RECTCROP = 15
    CIRCLECROP = 16
    WIPELEFT = 17
    WIPERIGHT = 18
    SLIDEDOWN = 19
    SLIDEUP = 20
    SLIDELEFT = 21
    SLIDERIGHT = 22


@dataclass_json
@dataclass
class TransitionConfig:
    effect: TransitionEffect
    # transition duration in seconds.
    duration: float = 2.0


class VideoAugmenterByConcat(BaseVidgearFFMPEGAugmenter):
    def __init__(
        self,
        video_paths: List[str],
        src_video_path_index: int,
        transition: Optional[TransitionConfig] = None,
    ):
        assert len(video_paths) > 0, "Please provide at least one input video"
        assert all(
            pathmgr.exists(video_path) for video_path in video_paths
        ), "Invalid video path(s) provided"

        self.video_paths = [
            pathmgr.get_local_path(video_path) for video_path in video_paths
        ]
        self.src_video_path_index = src_video_path_index

        video_info = get_video_info(self.video_paths[src_video_path_index])

        self.height = ceil(video_info["height"] / 2) * 2
        self.width = ceil(video_info["width"] / 2) * 2

        self.sample_aspect_ratio = video_info.get(
            "sample_aspect_ratio", self.width / self.height
        )
        self.transition = transition

    def _create_null_transition_filters(
        self,
        video_streams: List[str],
        audio_streams: List[str],
    ) -> List[str]:
        # Interleave the video and audio streams.
        all_streams = [v for pair in zip(video_streams, audio_streams) for v in pair]
        filters = [
            f"{''.join(all_streams)}concat=n={len(self.video_paths)}:v=1:a=1[v][a]"
        ]
        return filters

    def _create_transition_filters(
        self,
        video_streams: List[str],
        audio_streams: List[str],
        out_video: str = "[v]",
        out_audio: str = "[a]",
    ) -> List[str]:
        if self.transition is None:
            return self._create_null_transition_filters(video_streams, audio_streams)

        transition = self.transition
        effect = transition.effect.name.lower()

        video_durations = [
            float(get_video_info(video_path)["duration"])
            for video_path in self.video_paths
        ]
        log.info(f"Video durations = {video_durations}.")
        # There are 2 steps:
        # 1. Harmonize the timebase between clips;
        # 2. Add the transition filter.
        td = transition.duration
        concat_filters = []
        for i, name in enumerate(video_streams):
            fps_filter = f"[{i}fps]"
            concat_filters.append(f"{name}settb=AVTB,fps=30/1{fps_filter}")

        if td > min(video_durations):
            # Decrease transition duration (with padding) to prevent hung ffmpeg calls.
            new_td = max(0, min(video_durations) - 0.1)
            if new_td < 0.5:
                log.warn("Disabling transitions due to low duration: %f", new_td)
                return self._create_null_transition_filters(
                    video_streams, audio_streams
                )

            log.info(
                f"Transition duration {td} > {min(video_durations)}. Decreasing to {new_td}."
            )
            td = new_td

        prev = "[0fps]"
        cum_dur = video_durations[0]
        for i in range(1, len(video_durations) - 1):
            dur = video_durations[i]
            fps_filter = f"[{i}fps]"
            out_filter = f"[{i}m]"
            offset = cum_dur - td
            concat_filters.append(
                f"{prev}{fps_filter}xfade=transition={effect}:duration={td}:offset={offset}{out_filter}"
            )
            prev = out_filter
            cum_dur += dur - td

        # Special processing for the last filter to comply with out_video requirement.
        concat_filters.append(
            f"{prev}[{len(video_durations) - 1}fps]xfade=transition={effect}:duration={td}:offset={cum_dur - td}{out_video}"
        )

        # Concat audio filters.
        prev = audio_streams[0]
        cum_dur = video_durations[0]
        for i in range(1, len(video_durations) - 1):
            dur = video_durations[i]
            in_f = audio_streams[i]
            out_f = f"[a{i}m]"
            offset = cum_dur - td
            concat_filters.append(f"{prev}{in_f}acrossfade=d={td}:c1=tri:c2=tri{out_f}")
            prev = out_f
            cum_dur += dur - td

        concat_filters.append(
            f"{prev}[{len(video_durations) - 1}:a]acrossfade=d={td}:c1=tri:c2=tri{out_audio}"
        )

        return concat_filters

    def get_command(self, video_path: str, output_path: str) -> List[str]:
        """
        Concatenates multiple videos together

        @param video_path: the path to the video to be augmented

        @param output_path: the path in which the resulting video will be stored.

        @returns: a list of strings containing the CLI FFMPEG command for
            the augmentation
        """
        inputs = [["-i", video] for video in self.video_paths]
        flat_inputs = [element for sublist in inputs for element in sublist]
        filters = []
        video_streams = []
        audio_streams = []
        for i in range(len(self.video_paths)):
            filters.append(
                f"[{i}:v]scale={self.width}:{self.height}[{i}v],[{i}v]setsar=ratio="
                f"{self.sample_aspect_ratio}[{i}vf]"
            )
            video_streams.append(f"[{i}vf]")
            audio_streams.append(f"[{i}:a]")

        filters += self._create_transition_filters(video_streams, audio_streams)

        return [
            "-y",
            *flat_inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-vsync",
            "2",
            *self.output_fmt(output_path),
        ]
