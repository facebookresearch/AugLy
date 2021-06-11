#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import io
import math
import os
import shutil
from typing import Any, Dict, Optional, Union

import ffmpeg
from ffmpeg.nodes import FilterableStream
from augly.utils import pathmgr, SILENT_AUDIO_PATH
from augly.utils.ffmpeg import FFMPEG_PATH, FFPROBE_PATH


def combine_frames_and_audio_to_file(
    raw_frames: str,
    audio: Optional[Union[str, io.BytesIO]],
    output_path: str,
    framerate: float,
) -> None:
    frame_dir = os.path.dirname(raw_frames)
    if not os.path.isdir(frame_dir):
        raise RuntimeError(
            f"Got raw frames glob path of {raw_frames}, but {frame_dir} is not "
            "a directory"
        )

    video_stream = ffmpeg.input(raw_frames, pattern_type="glob", framerate=framerate)
    video_stream = video_stream.filter(
        "pad", **{"width": "ceil(iw/2)*2", "height": "ceil(ih/2)*2"}
    )
    merge_video_and_audio(video_stream, audio, output_path)


def extract_audio_to_file(video_path: str, output_audio_path: str) -> None:
    audio_info = get_audio_info(video_path)

    (
        ffmpeg.input(video_path, loglevel="quiet")
        .output(output_audio_path, acodec=audio_info["codec_name"], ac=1)
        .overwrite_output()
        .run(cmd=FFMPEG_PATH)
    )


def extract_frames_to_dir(
    video_path: str,
    output_dir: str,
    output_pattern: str = "raw_frame%08d.jpg",
    quality: int = 0,
    scale: float = 1,
) -> None:
    video_info = get_video_info(video_path)

    (
        ffmpeg.input(video_path, ss=0, loglevel="quiet")
        .filter("scale", f"iw*{scale}", f"ih*{scale}")
        .output(
            os.path.join(output_dir, output_pattern),
            vframes=video_info["nb_frames"],
            **{"qscale:v": quality},
        )
        .overwrite_output()
        .run(cmd=FFMPEG_PATH)
    )


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
    video_stream: FilterableStream,
    audio: Optional[Union[str, io.BytesIO]],
    output_path: str,
) -> None:
    kwargs = {"c:v": "libx264", "c:a": "copy", "bsf:a": "aac_adtstoasc"}
    if audio:
        audio_stream = ffmpeg.input(audio, loglevel="quiet")
        output = ffmpeg.output(
            video_stream, audio_stream, output_path, pix_fmt="yuv420p", **kwargs
        ).overwrite_output()
    else:
        output = ffmpeg.output(
            video_stream, output_path, pix_fmt="yuv420p", **kwargs
        ).overwrite_output()

    output.run(cmd=FFMPEG_PATH)
