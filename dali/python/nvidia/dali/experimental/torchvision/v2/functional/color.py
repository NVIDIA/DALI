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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali as dali

from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input, get_HWC_from_layout_dynamic
from ..color import Grayscale


def _grayscale(
    inpt: TensorLike | ndd.Batch,
    num_output_channels: int = 1,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:

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
    else:
        return ndd.hsv(inpt, saturation=0, device=device)


@adjust_input
def to_grayscale(
    inpt: TensorLike | ndd.Batch,
    num_output_channels: int = 1,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(inpt, num_output_channels, device)


@adjust_input
def rgb_to_grayscale(
    inpt: TensorLike | ndd.Batch,
    num_output_channels: int = 1,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    """
    Please refer to the ``Grayscale`` operator for more details.
    """
    return _grayscale(inpt, num_output_channels, device)
