#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from augly import utils
from augly.text import intensity as txtintensity


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
    texts: Optional[List[str]] = None,
    aug_texts: Optional[Union[List[str], str]] = None,
    **kwargs,
) -> None:
    if metadata is None:
        return

    assert isinstance(
        metadata, list
    ), "Expected `metadata` to be set to None or of type list"
    assert (
        texts is not None
    ), "Expected `texts` to be passed in if metadata was provided"
    assert (
        aug_texts is not None
    ), "Expected `aug_texts` to be passed in if metadata was provided"

    metadata.append(
        {
            "name": function_name,
            "input_type": "list" if isinstance(texts, list) else "string",
            "src_length": len(texts) if isinstance(texts, list) else 1,
            "dst_length": len(aug_texts) if isinstance(aug_texts, list) else 1,
            **kwargs,
        }
    )

    intensity_kwargs = {"metadata": metadata[-1], "texts": texts, **kwargs}
    metadata[-1]["intensity"] = getattr(
        txtintensity, f"{function_name}_intensity", lambda **_: 0.0
    )(**intensity_kwargs)


def get_gendered_words_mapping(mapping: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Note: The `swap_gendered_words` augmentation, including this logic, was originally
    written by Adina Williams and has been used in influential work, e.g.
    https://arxiv.org/pdf/2005.00614.pdf
    """
    assert isinstance(
        mapping, (str, Dict)
    ), "Mapping must be either a dict or filepath to a mapping of gendered words"

    if isinstance(mapping, Dict):
        return mapping

    if isinstance(mapping, str):
        with utils.pathmgr.open(mapping) as f:
            return json.load(f)
