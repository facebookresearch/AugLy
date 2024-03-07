#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import ffmpeg
import numpy as np
from augly.audio import utils as audutils
from augly.utils import pathmgr, SILENT_AUDIO_PATH
from augly.utils.ffmpeg import FFMPEG_PATH, FFPROBE_PATH
from vidgear.gears import WriteGear


def combine_frames_and_audio_to_file(
    raw_frames: str,
    audio: Optional[str],
    output_path: str,
    framerate: float,
) -> None:
    frame_dir = os.path.dirname(raw_frames)
    if not os.path.isdir(frame_dir):
        raise RuntimeError(
            f"Got raw frames glob path of {raw_frames}, but {frame_dir} is not "
            "a directory"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video_path = os.path.join(tmpdir, "out.mp4")
        ffmpeg_command = [
            "-y",
            "-framerate",
            str(framerate),
            "-pattern_type",
            "glob",
            "-i",
            raw_frames,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            temp_video_path,
        ]
        execute_vidgear_command(temp_video_path, ffmpeg_command)
        temp_padded_video_path = os.path.join(tmpdir, "out1.mp4")
        ffmpeg_command = [
            "-y",
            "-i",
            temp_video_path,
            "-vf",
            "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
            "-preset",
            "ultrafast",
            temp_padded_video_path,
        ]
        execute_vidgear_command(temp_padded_video_path, ffmpeg_command)
        merge_video_and_audio(temp_padded_video_path, audio, output_path)


def execute_vidgear_command(output_path: str, ffmpeg_command: List[str]) -> None:
    writer = WriteGear(output=output_path, logging=True)
    writer.execute_ffmpeg_cmd(ffmpeg_command)
    writer.close()


def extract_audio_to_file(video_path: str, output_audio_path: str) -> None:
    audio_info = get_audio_info(video_path)
    sample_rate = str(audio_info["sample_rate"])
    codec = audio_info["codec_name"]

    if os.path.splitext(output_audio_path)[-1] == ".aac":
        (
            ffmpeg.input(video_path, loglevel="quiet")
            .output(output_audio_path, acodec=codec, ac=1)
            .overwrite_output()
            .run(cmd=FFMPEG_PATH)
        )
    else:
        out, err = (
            ffmpeg.input(video_path, loglevel="quiet")
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
            .run(cmd=FFMPEG_PATH, capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        audutils.ret_and_save_audio(audio, output_audio_path, int(sample_rate))


def extract_frames_to_dir(
    video_path: str,
    output_dir: str,
    output_pattern: str = "raw_frame%08d.jpg",
    quality: int = 0,
    scale: float = 1,
) -> None:
    video_info = get_video_info(video_path)

    ffmpeg_command = [
        "-y",
        "-i",
        video_path,
        "-vf",
        f"scale=iw*{scale}:ih*{scale}",
        "-vframes",
        str(video_info["nb_frames"]),
        "-qscale:v",
        str(quality),
        "-preset",
        "ultrafast",
        os.path.join(output_dir, output_pattern),
    ]
    execute_vidgear_command(os.path.join(output_dir, output_pattern), ffmpeg_command)


def get_audio_info(media_path: str) -> Dict[str, Any]:
    """
    Returns whatever ffprobe returns. Of particular use are things such as the
    encoder ("codec_name") used for audio encoding, the sample rate ("sample_rate"),
    and length in seconds ("duration")

    Accepts as input either an audio or video path.
    """
    try:
        local_media_path = pathmgr.get_local_path(media_path)
    except RuntimeError:
        raise FileNotFoundError(f"Provided media path {media_path} does not exist")

    probe = ffmpeg.probe(local_media_path, cmd=FFPROBE_PATH)
    audio_info = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
        None,
    )

    assert (
        audio_info is not None
    ), "Error retrieving audio metadata, please verify that an audio stream exists"

    return audio_info


def get_video_fps(video_path: str) -> Optional[float]:
    video_info = get_video_info(video_path)

    try:
        frame_rate = video_info["avg_frame_rate"]
        # ffmpeg often returns fractional framerates, e.g. 225480/7523
        if "/" in frame_rate:
            num, denom = (float(f) for f in frame_rate.split("/"))
            return num / denom
        else:
            return float(frame_rate)
    except Exception:
        return None


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Returns whatever ffprobe returns. Of particular use are things such as the FPS
    ("avg_frame_rate"), number of raw frames ("nb_frames"), height and width of each
    frame ("height", "width") and length in seconds ("duration")
    """
    try:
        local_video_path = pathmgr.get_local_path(video_path)
    except RuntimeError:
        raise FileNotFoundError(f"Provided video path {video_path} does not exist")

    probe = ffmpeg.probe(local_video_path, cmd=FFPROBE_PATH)
    video_info = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )

    assert (
        video_info is not None
    ), "Error retrieving video metadata, please verify that the video file exists"

    return video_info


def has_audio_stream(video_path: str) -> bool:
    streams = ffmpeg.probe(video_path, cmd=FFPROBE_PATH)["streams"]
    for stream in streams:
        if stream["codec_type"] == "audio":
            return True
    return False


def add_silent_audio(
    video_path: str,
    output_path: Optional[str] = None,
    duration: Optional[float] = None,
) -> None:
    local_video_path = pathmgr.get_local_path(video_path)
    if local_video_path != video_path:
        assert (
            output_path is not None
        ), "If remote video_path is provided, an output_path must be provided"
        video_path = local_video_path
    output_path = output_path or video_path

    if has_audio_stream(video_path):
        if video_path != output_path:
            shutil.copy(video_path, output_path)
        return

    duration = duration or float(get_video_info(video_path)["duration"])
    video = ffmpeg.input(video_path).video
    silent_audio_path = pathmgr.get_local_path(SILENT_AUDIO_PATH)
    audio = ffmpeg.input(silent_audio_path, stream_loop=math.ceil(duration)).audio
    output = ffmpeg.output(video, audio, output_path, pix_fmt="yuv420p", t=duration)
    output.overwrite_output().run(cmd=FFMPEG_PATH)


def merge_video_and_audio(
    video_path: str,
    audio_path: Optional[str],
    output_path: str,
) -> None:
    ffmpeg_command = []

    if audio_path:
        ffmpeg_command = [
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-vf",
            "format=pix_fmts=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-preset",
            "ultrafast",
            output_path,
        ]
    else:
        ffmpeg_command = [
            "-y",
            "-i",
            video_path,
            "-vf",
            "format=pix_fmts=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-preset",
            "ultrafast",
            output_path,
        ]

    execute_vidgear_command(output_path, ffmpeg_command)
