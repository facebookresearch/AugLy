#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import setuptools


requirements = [
    r for r in Path("requirements.txt").read_text().splitlines() if "@" not in r
]

extra_requirements = {
    "av": [
        r for r in Path("av_requirements.txt").read_text().splitlines() if "@" not in r
    ]
}

with open("README.md", encoding="utf8") as f:
    readme = f.read()


setuptools.setup(
    name="augly",
    version=open("version.txt", "r").read(),
    description="A data augmentations library for audio, image, text, & video.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/AugLy",
    author="Zoe Papakipos and Joanna Bitton",
    author_email="zoep@fb.com",
    packages=setuptools.find_packages(exclude=["augly.tests"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extra_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
