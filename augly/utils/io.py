#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from iopath.common.file_io import HTTPURLHandler, PathManager


pathmgr = PathManager()
pathmgr.register_handler(HTTPURLHandler())
pathmgr.set_strict_kwargs_checking(False)
