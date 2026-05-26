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

import operator

import nvidia.dali.experimental.dynamic as ndd
from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input
from ..randomcrop import RandomCrop


def _validate_integer_param(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as err:
        raise TypeError(f"{name} must be an integer, got {type(value)}") from err


def _round_pil_box(top, left, height, width) -> tuple[int, int, int, int]:
    try:
        rounded_top = int(round(top))
        rounded_left = int(round(left))
        rounded_bottom = int(round(top + height))
        rounded_right = int(round(left + width))
    except TypeError as err:
        raise TypeError("top, left, height, and width must be real numbers") from err

    return (
        rounded_top,
        rounded_left,
        rounded_bottom - rounded_top,
        rounded_right - rounded_left,
    )


def _is_pil_image_layout(inpt: TensorLike | ndd.Batch) -> bool:
    return inpt.layout[-3:] == "HWC"


def _validate_crop_params(inpt, top, left, height, width) -> tuple[int, int, int, int]:
    if _is_pil_image_layout(inpt):
        return _round_pil_box(top, left, height, width)
    return (
        _validate_integer_param(top, "top"),
        _validate_integer_param(left, "left"),
        _validate_integer_param(height, "height"),
        _validate_integer_param(width, "width"),
    )


def _verify_crop_coordinate(value, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value)}")
def _validate_integer_param(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as err:
        raise TypeError(f"{name} must be an integer, got {type(value)}") from err


def _round_pil_box(top, left, height, width) -> tuple[int, int, int, int]:
    try:
        rounded_top = int(round(top))
        rounded_left = int(round(left))
        rounded_bottom = int(round(top + height))
        rounded_right = int(round(left + width))
    except TypeError as err:
        raise TypeError("top, left, height, and width must be real numbers") from err

    return (
        rounded_top,
        rounded_left,
        rounded_bottom - rounded_top,
        rounded_right - rounded_left,
    )


def _is_pil_image_layout(inpt: TensorLike | ndd.Batch) -> bool:
    return inpt.layout[-3:] == "HWC"


def _validate_crop_params(inpt, top, left, height, width) -> tuple[int, int, int, int]:
    if _is_pil_image_layout(inpt):
        return _round_pil_box(top, left, height, width)
    return (
        _validate_integer_param(top, "top"),
        _validate_integer_param(left, "left"),
        _validate_integer_param(height, "height"),
        _validate_integer_param(width, "width"),
    )


@adjust_input
def crop(
    inpt: TensorLike | ndd.Batch,
    top: int | float,
    left: int | float,
    height: int | float,
    width: int | float,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    """
    Please refer to the ``RandomCrop`` operator for more details.
    """
    top, left, height, width = _validate_crop_params(inpt, top, left, height, width)
    RandomCrop.verify_args(
        size=(height, width),
        padding=None,
        pad_if_needed=False,
        padding_mode="constant",
        fill=0,
    )

    return ndd.slice(
        inpt,
        [float(left), float(top)],
        [float(width), float(height)],
        normalized_anchor=False,
        normalized_shape=False,
        out_of_bounds_policy="pad",
        fill_values=0,
        device=device,
    )
