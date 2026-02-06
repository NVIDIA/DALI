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
from torch import Tensor
import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali as dali

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..color import Grayscale  # noqa: E402


def _grayscale(
    img: Tensor,
    num_output_channels: int = 1,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Tensor:

    Grayscale.verify_args(num_output_channels=num_output_channels)

    c = img.shape[-1]
    if num_output_channels == 1 and c == 3:  # RGB (TODO: what if it is HSV?)
        return ndd.color_space_conversion(
            img,
            image_type=dali.types.RGB,
            output_type=dali.types.GRAY,
            device=device,
        )
    else:
        return ndd.hsv(img, saturation=0, device=device)


@adjust_input
def to_grayscale(
    img: Tensor,
    num_output_channels: int = 1,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Tensor:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(img, num_output_channels, device)


@adjust_input
def rgb_to_grayscale(
    img: Tensor, num_output_channels: int = 1, device: Literal["cpu", "gpu"] = "cpu"
) -> Tensor:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(img, num_output_channels, device)
