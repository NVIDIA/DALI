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
from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input
from ..randomcrop import RandomCrop


def _get_crop_axes(inpt: TensorLike | ndd.Batch) -> list[int]:
    layout = inpt.layout[-3:]
    if layout == "HWC":
        return [-3, -2]
    if layout == "CHW":
        return [-2, -1]
    if inpt.layout[-2:] == "HW":
        return [-2, -1]
    raise ValueError(f"Unsupported layout: {inpt.layout!r}. Expected one of HWC, CHW, HW.")


def _verify_crop_coordinate(value, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)}")


@adjust_input
def crop(
    inpt: TensorLike | ndd.Batch,
    top: int,
    left: int,
    height: int,
    width: int,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    """
    Please refer to the ``RandomCrop`` operator for more details.
    """
    _verify_crop_coordinate(top, "top")
    _verify_crop_coordinate(left, "left")
    RandomCrop.verify_args(
        size=(height, width),
        padding=None,
        pad_if_needed=False,
        padding_mode="constant",
        fill=0,
    )

    return ndd.slice(
        inpt,
        (top, left),
        (height, width),
        axes=_get_crop_axes(inpt),
        out_of_bounds_policy="pad",
        fill_values=0,
        device=device,
    )
