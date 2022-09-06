#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional


class Segment(NamedTuple):
    start: float
    end: float
    src_id: Optional[str] = None

    def delta(self, start_delta: float, end_delta: float):
        new_start = self.start + start_delta
        new_end = self.end + end_delta
        if new_start > new_end:
            raise ValueError(
                f"Invalid segment created: expected {new_start} < {new_end}."
            )

        return Segment(new_start, new_end, self.src_id)
