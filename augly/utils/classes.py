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
        return Segment(self.start + start_delta, self.end + end_delta, self.src_id)
