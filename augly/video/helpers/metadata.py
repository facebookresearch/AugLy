#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import augly.utils as utils
import augly.video.helpers as helpers
import augly.video.helpers.intensity as vdintensity


Segment = utils.Segment


def get_func_kwargs(
    metadata: Optional[List[Dict[str, Any]]],
    local_kwargs: Dict[str, Any],
    video_path: str,
    **kwargs,
) -> Dict[str, Any]:
    if metadata is None:
        return {}
    func_kwargs = deepcopy(local_kwargs)
    func_kwargs.pop("metadata")
    func_kwargs.update(
        {
            "src_video_info": helpers.get_video_info(video_path),
            "src_fps": helpers.get_video_fps(video_path),
            **kwargs,
        }
    )
    return func_kwargs


def compute_time_crop_segments(
    src_segment: Segment,
    dst_segment: Segment,
    speed_factor: float,
    crop_start: float,
    crop_end: float,
    new_src_segments: List[Segment],
    new_dst_segments: List[Segment],
) -> None:
    """
    Calculates how the given matching pair src_segment & dst_segment change
    given a temporal crop starting at crop_start & ending at crop_end. We can
    use the same logic here for multiple transforms, by setting the crop_start
    & crop_end depending on the transform kwargs.

    Doesn't return anything, but appends the new matching segments in the dst
    video corresponding to the pair passed in to new_src_segments & new_dst_segments,
    if the segment pair still matches in the dst video. If the passed in segment
    pair is cropped out as a result of this temporal crop, nothing will be
    appended to the lists, since the segment pair doesn't exist in the dst video.
    """
    # Crop segment is outside of the initial clip, so this matching segment
    # pair no longer exists in the new video.
    if crop_start >= dst_segment.end or crop_end <= dst_segment.start:
        return
    # new_start represents where the matching segment starts in the dst audio
    # (if negative, then part of the matching segment is getting cut out, so
    # we need to adjust both the src & dst starts).
    new_start = dst_segment.start - crop_start
    src_start, src_end = src_segment
    if new_start < 0:
        # We're cropping the beginning of the matching segment.
        # Note: if the video was sped up before, we need to take this into account
        # (the matching segment that is e.g. 10 seconds of dst audio might
        # correspond to 5 seconds of src audio, if it was previously
        # slowed down by 0.5x).
        src_start = src_segment.start - new_start * speed_factor
        new_start = 0
    new_end = min(dst_segment.end - crop_start, crop_end - crop_start)
    if crop_end < dst_segment.end:
        # We're cropping the end of the matching segment.
        # Note: if the video was sped up before, we need to take this into
        # account (as above).
        src_end = src_segment.end - (dst_segment.end - crop_end) * speed_factor

    new_src_segments.append(Segment(src_start, src_end))
    new_dst_segments.append(Segment(new_start, new_end))


def compute_time_decimate_segments(
    src_segment: Segment,
    dst_segment: Segment,
    src_duration: float,
    speed_factor: float,
    new_src_segments: List[Segment],
    new_dst_segments: List[Segment],
    **kwargs,
) -> None:
    on_segment = src_duration * kwargs["on_factor"]
    off_segment = on_segment * kwargs["off_factor"]
    n = int(src_duration / (on_segment + off_segment))

    for i in range(n):
        crop_start = i * on_segment + i * off_segment
        crop_end = min(src_duration, (i + 1) * on_segment + i * off_segment)

        if crop_start > src_duration:
            break

        compute_time_crop_segments(
            src_segment,
            dst_segment,
            speed_factor,
            crop_start,
            crop_end,
            new_src_segments,
            new_dst_segments,
        )

    for i in range(1, len(new_dst_segments)):
        new_start = new_dst_segments[i].start + new_dst_segments[i - 1].end
        new_dst_segments[i] = Segment(
            new_start,
            new_start + new_dst_segments[i].end,
        )


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
            # Note: When we implement insert_in_background, make sure to pass these kwargs
            offset = kwargs["offset_factor"] * kwargs["background_video_duration"]
            # The matching segments are just offset in the dst audio by the amount
            # of background video inserted before the src video.
            new_src_segments.append(src_segment)
            new_dst_segments.append(
                Segment(dst_segment.start + offset, dst_segment.end + offset)
            )
        elif name == "replace_with_background":
            new_src_segments.append(src_segment)
            clip_start = kwargs["starting_background_duration"]
            duration = kwargs["source_duration"]
            new_dst_segments.append(
                Segment(dst_segment.start + clip_start, dst_segment.start + clip_start + duration)
            )
        elif name == "change_video_speed":
            # speed_factor > 1 if speedup, < 1 if slow down
            speed_factor *= src_duration / dst_duration
            new_src_segments.append(src_segment)
            new_dst_segments.append(
                Segment(
                    dst_segment.start / speed_factor, dst_segment.end / speed_factor
                )
            )
        elif name == "concat":
            new_src_segments.append(src_segment)
            offset = sum(
                float(helpers.get_video_info(vp)["duration"])
                for vp in kwargs["video_paths"][: kwargs["src_video_path_index"]]
            )
            new_dst_segments.append(
                Segment(
                    dst_segment.start + offset,
                    dst_segment.end + offset,
                )
            )
        elif name == "loop":
            # The existing segments are unchanged.
            new_src_segments.append(src_segment)
            new_dst_segments.append(dst_segment)
            # Each original src segments now additionally matches a segment in
            # each loop in the dst video.
            for l_idx in range(kwargs["num_loops"]):
                new_src_segments.append(src_segment)
                new_dst_segments.append(
                    Segment(
                        dst_segment.start + (l_idx + 1) * src_duration,
                        dst_segment.end + (l_idx + 1) * src_duration,
                    )
                )
        elif name == "time_crop":
            crop_start = kwargs["offset_factor"] * src_duration
            crop_end = crop_start + kwargs["duration_factor"] * src_duration
            compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                crop_start,
                crop_end,
                new_src_segments,
                new_dst_segments,
            )
        elif name == "time_decimate":
            compute_time_decimate_segments(
                src_segment,
                dst_segment,
                src_duration,
                speed_factor,
                new_src_segments,
                new_dst_segments,
                **kwargs,
            )
        elif name == "trim":
            crop_start = kwargs["start"] or 0.0
            crop_end = kwargs["end"] or src_duration
            compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                crop_start,
                crop_end,
                new_src_segments,
                new_dst_segments,
            )
        elif name == "replace_with_color_frames":
            # This transform is like the inverse of time_crop/trim, because
            # offset & duration denote where the src content is being cropped
            # out, instead of which src content is being kept.
            offset = kwargs["offset_factor"] * src_duration
            duration = kwargs["duration_factor"] * src_duration
            compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                0.0,
                offset,
                new_src_segments,
                new_dst_segments,
            )
            compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                offset + duration,
                dst_duration,
                new_src_segments,
                new_dst_segments,
            )
    return new_src_segments, new_dst_segments


def compute_segments(
    name: str,
    src_duration: float,
    dst_duration: float,
    metadata: List[Dict[str, Any]],
    **kwargs,
) -> Tuple[List[Segment], List[Segment]]:
    """
    Compute matching pairs of src_segment -> dst_segment, given the kwargs of the
    transform, as well as the metadata about previously applied transforms.
    """
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
            if meta["name"] in ["change_video_speed"]:
                speed_factor *= meta["factor"]

    if name in [
        "insert_in_background",
        "replace_with_background",
        "change_video_speed",
        "loop",
        "time_crop",
        "time_decimate",
        "trim",
        "replace_with_color_frames",
        "concat",
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


def get_metadata(
    metadata: Optional[List[Dict[str, Any]]],
    function_name: str,
    video_path: str,
    output_path: Optional[str],
    src_video_info: Dict[str, Any],
    src_fps: Optional[float],
    **kwargs,
) -> None:
    if metadata is None:
        return

    assert isinstance(
        metadata, list
    ), "Expected 'metadata' to be set to None or of type list"

    assert src_fps is not None

    output_path = output_path or video_path

    src_video_info = src_video_info
    dst_video_info = helpers.get_video_info(output_path)

    src_duration = float(src_video_info["duration"])
    dst_duration = float(dst_video_info["duration"])
    src_segments, dst_segments = compute_segments(
        function_name, src_duration, dst_duration, metadata, **kwargs
    )

    # Json can't represent tuples, so they're represented as lists, which should
    # be equivalent to tuples. So let's avoid tuples in the metadata by
    # converting any tuples to lists here.
    kwargs_types_fixed = dict(
        (k, list(v)) if isinstance(v, tuple) else (k, v) for k, v in kwargs.items()
    )

    metadata.append(
        {
            "name": function_name,
            "src_duration": src_duration,
            "dst_duration": dst_duration,
            "src_fps": src_fps,
            "dst_fps": helpers.get_video_fps(output_path),
            "src_width": src_video_info["width"],
            "src_height": src_video_info["height"],
            "dst_width": dst_video_info["width"],
            "dst_height": dst_video_info["height"],
            "src_segments": [src_segment._asdict() for src_segment in src_segments],
            "dst_segments": [dst_segment._asdict() for dst_segment in dst_segments],
            **kwargs_types_fixed,
        }
    )

    intensity_kwargs = {"metadata": metadata[-1], **kwargs}
    metadata[-1]["intensity"] = getattr(
        vdintensity, f"{function_name}_intensity", lambda **_: 0.0
    )(**intensity_kwargs)
