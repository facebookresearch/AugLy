#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import setuptools
from pathlib import Path
from typing import Dict


requirements = [
    r
    for r in Path(f"requirements.txt").read_text().splitlines()
    if '@' not in r
]


setuptools.setup(
    name="augly",
    version="1.0.0",
    description="A data augmentations library for audio, image, text, & video.",
    long_description=open("README.md").read(),
    url="https://github.com/facebookresearch/AugLy",
    author="zpapakipos",
    author_email="zoep@fb.com",
    packages=["augly"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.7",
)
