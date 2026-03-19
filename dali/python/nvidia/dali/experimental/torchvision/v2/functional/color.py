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

from typing import Literal
import torch
from PIL import Image
import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali as dali

from ..operator import adjust_input, get_HWC_from_layout_dynamic  # noqa: E402
from ..color import Grayscale  # noqa: E402


def _grayscale(
    inpt: Image.Image | torch.Tensor,
    num_output_channels: int = 1,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:

    Grayscale.verify_args(num_output_channels=num_output_channels)

    _, _, c = get_HWC_from_layout_dynamic(inpt)
    if num_output_channels == 1 and c == 3:  # RGB (TODO: what if it is HSV?)
        return ndd.color_space_conversion(
            inpt,
            image_type=dali.types.RGB,
            output_type=dali.types.GRAY,
            device=device,
        )
    elif num_output_channels == 1 and c == 1:
        return inpt
    elif num_output_channels == 3 and c == 1:
        return ndd.cat(inpt, inpt, inpt, axis_name="C")
    else:
        return ndd.hsv(inpt, saturation=0, device=device)


@adjust_input
def to_grayscale(
    inpt: Image.Image | torch.Tensor,
    num_output_channels: int = 1,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(inpt, num_output_channels, device)

@adjust_input
def rgb_to_grayscale(
    inpt: Image.Image | torch.Tensor,
    num_output_channels: int = 1,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(inpt, num_output_channels, device)
