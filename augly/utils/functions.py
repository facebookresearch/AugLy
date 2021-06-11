#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

from augly.utils.classes import Segment


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
