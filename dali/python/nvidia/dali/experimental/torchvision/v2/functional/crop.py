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
from typing import List

import nvidia.dali.experimental.dynamic as ndd
from torchvision.transforms import InterpolationMode

from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input
from ..randomcrop import RandomCrop
from ..resize import Resize


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


def _crop(
    inpt: ndd.Tensor | ndd.Batch,
    top: int,
    left: int,
    height: int,
    width: int,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    axes = [-3, -2] if _is_pil_image_layout(inpt) else [-2, -1]
    return ndd.slice(
        inpt,
        (top, left),
        (height, width),
        axes=axes,
        out_of_bounds_policy="pad",
        fill_values=0,
        device=device,
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

    return _crop(inpt, top, left, height, width, device=device)


@adjust_input
def resized_crop(
    inpt: TensorLike | ndd.Batch,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | List[int],
    interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
    antialias: bool = True,
    device: DeviceLike = "cpu",
) -> ndd.Tensor | ndd.Batch:
    """
    Crop the input at location (top, left) with dimensions (height, width),
    then resize the crop to the given size.
    """
    top, left, height, width = _validate_crop_params(inpt, top, left, height, width)
    RandomCrop.verify_args(
        size=(height, width),
        padding=None,
        pad_if_needed=False,
        padding_mode="constant",
        fill=0,
    )
    interpolation = Resize.normalize_interpolation(interpolation)
    Resize.verify_args(size=size, max_size=None, interpolation=interpolation, antialias=antialias)

    size_normalized = Resize.infer_effective_size(size)
    interpolation = Resize.interpolation_modes[interpolation]

    cropped = _crop(inpt, top, left, height, width, device=device)
    target_h, target_w = Resize.calculate_target_size_dynamic_mode(
        (height, width), size_normalized, None
    )

    return ndd.resize(
        cropped,
        device=device,
        size=(target_h, target_w),
        interp_type=interpolation,
        antialias=antialias,
    )
