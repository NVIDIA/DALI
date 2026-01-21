# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Literal
import nvidia.dali.experimental.dynamic as ndd

import sys

import torch
from PIL import Image

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..pad import PadBase, PADDING_CLASS  # noqa: E402


@adjust_input
def _pad(
    img: ndd.Tensor | ndd.Batch,
    padding: List[int],
    fill: int | float = 0,
    axes: List[int] = [-3, -2],
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    device: Literal["cpu", "gpu"] = "cpu",
) -> ndd.Tensor:

    left, top, right, bottom = PadBase.get_padding(padding)
    # TODO: need to diversify based on layout
    # batch shape
    if isinstance(img, ndd.Tensor):
        input_height, input_width = (img.shape[-2], img.shape[-1])
    elif isinstance(img, ndd.Batch):
        # TODO: what in case of ragged batches?
        input_height, input_width = (img.shape[0][-2], img.shape[0][-1])
    else:
        raise TypeError(f"Type not supported {type(img)}")

    border_type = PADDING_CLASS[padding_mode].border_type

    return ndd.slice(
        img,
        (-top, -left),  # __anchor
        (input_height + bottom + top, input_width + right + left),  # __shape
        out_of_bounds_policy=border_type,
        fill_values=fill,
        axes=axes,
        device=device,
    )


def pad(
    img: Image.Image | torch.Tensor,
    padding: List[int],
    fill: int | float = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    device: Literal["cpu", "gpu"] = "cpu",
) -> ndd.Tensor:
    """
    Please refer to the ``Pad`` operator for more details.
    """
    if isinstance(img, Image.Image):
        axes = [-3, -2]
    elif isinstance(img, torch.Tensor):
        axes = [-2, -1]
    else:
        raise TypeError(f"Type not supported {type(img)}")

    return _pad(
        img,
        padding=padding,
        fill=fill,
        axes=axes,
        padding_mode=padding_mode,
        device=device,
    )
