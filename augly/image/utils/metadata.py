#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Optional

import augly.image.intensity as imintensity
from PIL import Image


def get_func_kwargs(
    metadata: Optional[List[Dict[str, Any]]], local_kwargs: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    if metadata is None:
        return {}
    func_kwargs = deepcopy(local_kwargs)
    func_kwargs.pop("metadata")
    func_kwargs.update(**kwargs)
    return func_kwargs


def get_metadata(
    metadata: Optional[List[Dict[str, Any]]],
    function_name: str,
    image: Optional[Image.Image] = None,
    aug_image: Optional[Image.Image] = None,
    **kwargs,
) -> None:
    if metadata is None:
        return

    assert isinstance(
        metadata, list
    ), "Expected `metadata` to be set to None or of type list"
    assert (
        image is not None
    ), "Expected `image` to be passed in if metadata was provided"
    assert (
        aug_image is not None
    ), "Expected `aug_image` to be passed in if metadata was provided"

    # Json can't represent tuples, so they're represented as lists, which should
    # be equivalent to tuples. So let's avoid tuples in the metadata by
    # converting any tuples to lists here.
    kwargs_types_fixed = dict(
        (k, list(v)) if isinstance(v, tuple) else (k, v) for k, v in kwargs.items()
    )

    metadata.append(
        {
            "name": function_name,
            "src_width": image.width,
            "src_height": image.height,
            "dst_width": aug_image.width,
            "dst_height": aug_image.height,
            **kwargs_types_fixed,
        }
    )

    intensity_kwargs = {"metadata": metadata[-1], **kwargs}
    metadata[-1]["intensity"] = getattr(
        imintensity, f"{function_name}_intensity", lambda **_: 0.0
    )(**intensity_kwargs)
