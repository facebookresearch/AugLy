#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from iopath.common.file_io import HTTPURLHandler, PathManager


pathmgr = PathManager()
pathmgr.register_handler(HTTPURLHandler())
pathmgr.set_strict_kwargs_checking(False)
