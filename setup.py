#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

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
    version="0.1.10",
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
