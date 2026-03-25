# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from PIL import Image

from ..operator import adjust_input, get_HWC_from_layout_dynamic  # noqa: E402
from ..pad import _PadBase, PADDING_CLASS, _ValidatePaddingMode  # noqa: E402


@adjust_input
def _pad(
    inpt: ndd.Tensor | ndd.Batch,
    padding: List[int],
    fill: int | float = 0,
    axes: List[int] = [-3, -2],
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    device: Literal["cpu", "gpu"] = "cpu",
) -> ndd.Tensor | ndd.Batch:

    left, top, right, bottom = _PadBase.get_padding(padding)
    input_height, input_width, _ = get_HWC_from_layout_dynamic(inpt)

    border_type = PADDING_CLASS[padding_mode].border_type

    return ndd.slice(
        inpt,
        (-top, -left),  # __anchor
        (input_height + bottom + top, input_width + right + left),  # __shape
        out_of_bounds_policy=border_type,
        fill_values=fill,
        axes=axes,
        device=device,
    )


def pad(
    inpt: Image.Image | torch.Tensor,
    padding: List[int],
    fill: int | float = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:
    """
    Please refer to the ``Pad`` operator for more details.
    """
    if isinstance(inpt, Image.Image):
        axes = [-3, -2]
    elif isinstance(inpt, torch.Tensor):
        axes = [-2, -1]
    else:
        raise TypeError(f"Type not supported {type(inpt)}")

    _ValidatePaddingMode.verify(padding_mode=padding_mode)

    return _pad(
        inpt,
        padding=padding,
        fill=fill,
        axes=axes,
        padding_mode=padding_mode,
        device=device,
    )
