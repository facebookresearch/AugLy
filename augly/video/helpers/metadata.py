#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from augly import utils
from augly.video import helpers
from augly.video.helpers import intensity as vdintensity


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
    end_dst_offset: float = 0.0,
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
    # Note: if the video was sped up before, we need to take this into account
    # (the matching segment that is e.g. 10 seconds of dst audio might
    # correspond to 5 seconds of src audio, if it was previously
    # slowed down by 0.5x).
    new_start = (dst_segment.start - crop_start) * speed_factor
    src_start, src_end, src_id = src_segment
    if new_start < 0:
        # We're cropping the beginning of the matching segment.
        src_start = src_segment.start - new_start
        new_start = 0
    new_end = min(dst_segment.end - crop_start, crop_end - crop_start)
    if crop_end < dst_segment.end:
        # We're cropping the end of the matching segment.
        # Note: if the video was sped up before, we need to take this into
        # account (as above).
        src_end = src_segment.end - (dst_segment.end - crop_end) * speed_factor

    new_src_segments.append(Segment(src_start, src_end, src_id))
    new_dst_segments.append(
        Segment(new_start + end_dst_offset, new_end + end_dst_offset)
    )


def compute_insert_in_background_multiple_segments(
    src_segment_starts: List[float],
    src_segment_ends: List[float],
    bkg_insertion_points: List[float],
    src_ids: List[str],
    transition_duration: float,
    new_src_segments: List[Segment],
    new_dst_segments: List[Segment],
    **kwargs,
) -> None:
    n = len(src_segment_starts)
    assert n == len(
        src_segment_ends
    ), "Source segment starts and ends lists must have equal length."
    assert n == len(
        bkg_insertion_points
    ), "Source segment starts and background insertion points lists must have equal length."
    assert n == len(
        src_ids
    ), "Source segment starts and source ids lists must have equal length."

    if n == 0:  # nothing to do
        return

    dst_cum_dur = 0.0  # background cumulative duration.
    offset = transition_duration / 2.0
    prev_bkg = 0.0
    for src_start, src_end, src_id, bkg_pt in zip(
        src_segment_starts, src_segment_ends, src_ids, bkg_insertion_points
    ):
        crop_start = src_start + offset
        crop_end = src_end - offset
        dst_start = dst_cum_dur + (bkg_pt - prev_bkg) - offset
        src_segment = Segment(start=crop_start, end=crop_end, src_id=src_id)
        dst_segment = Segment(start=dst_start, end=dst_start + (crop_end - crop_start))

        new_src_segments.append(src_segment)
        new_dst_segments.append(dst_segment)
        dst_cum_dur = dst_segment.end - offset
        prev_bkg = bkg_pt


def compute_time_decimate_segments(
    src_segment: Segment,
    dst_segment: Segment,
    src_duration: float,
    speed_factor: float,
    transition_duration: float,
    new_src_segments: List[Segment],
    new_dst_segments: List[Segment],
    **kwargs,
) -> None:
    start_offset = src_duration * kwargs["start_offset_factor"]
    on_segment = src_duration * kwargs["on_factor"]
    off_segment = on_segment * kwargs["off_factor"]
    n = int((src_duration - start_offset) / (on_segment + off_segment))

    dst_offset = 0
    for i in range(n):
        crop_start = (
            start_offset
            + i * on_segment
            + i * off_segment
            + (i > 0) * transition_duration / 2.0
        )
        crop_end = (
            start_offset
            + (i + 1) * on_segment
            + i * off_segment
            - (i < n - 1) * transition_duration / 2
        )
        crop_end = min(src_duration, crop_end)

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
            end_dst_offset=dst_offset,
        )
        dst_offset = new_dst_segments[-1].end


def get_transition_duration(kwargs):
    transition = kwargs.get("transition")
    if transition:
        return transition.duration

    return 0.0


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
    td = get_transition_duration(kwargs)
    for src_segment, dst_segment in zip(src_segments, dst_segments):
        if name == "insert_in_background":
            # Note: When we implement insert_in_background, make sure to pass these kwargs
            offset = kwargs["offset_factor"] * kwargs["background_video_duration"]
            transition_before = int(kwargs["transition_before"])
            transition_after = int(kwargs["transition_after"])
            # The matching segments are just offset in the dst audio by the amount
            # of background video inserted before the src video.
            new_src_segments.append(
                src_segment.delta(
                    transition_before * td / 2, -transition_after * td / 2
                )
            )
            new_dst_segments.append(
                Segment(
                    dst_segment.start + offset - transition_before * td / 2,
                    dst_segment.end
                    + offset
                    - transition_before * td
                    - transition_after * td / 2,
                )
            )
        elif name == "insert_in_background_multiple":
            compute_insert_in_background_multiple_segments(
                src_segment_starts=kwargs["src_segment_starts"],
                src_segment_ends=kwargs["src_segment_ends"],
                bkg_insertion_points=kwargs["bkg_insertion_points"],
                src_ids=kwargs["src_ids"],
                transition_duration=td,
                new_src_segments=new_src_segments,
                new_dst_segments=new_dst_segments,
            )
        elif name == "replace_with_background":
            clip_start = kwargs["starting_background_duration"]
            duration = kwargs["source_duration"]
            compute_time_crop_segments(
                src_segment,
                dst_segment,
                speed_factor,
                clip_start,
                clip_start + duration,
                new_src_segments,
                new_dst_segments,
                end_dst_offset=clip_start,
            )
        elif name == "change_video_speed":
            crt_factor = kwargs["factor"]
            global_factor = crt_factor * speed_factor
            new_src_segments.append(src_segment)
            new_dst_segments.append(
                Segment(
                    dst_segment.start / global_factor,
                    dst_segment.end / global_factor,
                )
            )
        elif name == "concat":
            src_index = kwargs["src_video_path_index"]
            num_videos = len(kwargs["video_paths"])
            transition_offset_start = td / 2 if src_index > 0 else 0.0
            transition_offset_end = td / 2 if src_index < num_videos - 1 else 0.0
            new_src_segments.append(
                src_segment.delta(transition_offset_start, -transition_offset_end)
            )
            offset = sum(
                float(helpers.get_video_info(vp)["duration"]) - td
                for vp in kwargs["video_paths"][: kwargs["src_video_path_index"]]
            )
            new_dst_segments.append(
                Segment(
                    dst_segment.start + offset + transition_offset_start,
                    dst_segment.end + offset - transition_offset_end,
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
                td,
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
    src_id = kwargs.get("src_id", None)
    if not metadata:
        src_segments = [Segment(0.0, src_duration, src_id)]
        dst_segments = [Segment(0.0, src_duration)]
    else:
        src_segments = [
            Segment(
                segment_dict["start"], segment_dict["end"], segment_dict.get("src_id")
            )
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
        "insert_in_background_multiple",
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
