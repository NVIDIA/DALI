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

import numbers
from typing import Literal, Sequence

import nvidia.dali as dali
import nvidia.dali.fn as fn

from ._enums import InterpolationMode
from .centercrop import CenterCrop
from .operator import (
    Operator,
    _ArgumentValidateRule,
    _ValidateIfNonNegative,
    _ValidateSizeDescriptor,
    get_HWC_from_layout_pipeline,
)
from .pad import PADDING_CLASS, _ValidatePaddingMode
from .resize import Resize


class _ValidateCropSize(_ArgumentValidateRule):
    """
    Verify RandomCrop size values.
    """

    @classmethod
    def verify(cls, *, size, **_) -> None:
        if isinstance(size, (list, tuple)) and any(not isinstance(value, int) for value in size):
            raise ValueError(f"Size values must be integers, got {size}")


class _ValidatePadding(_ArgumentValidateRule):
    """
    Verify RandomCrop padding arguments.
    """

    @classmethod
    def verify(cls, *, padding, pad_if_needed, padding_mode, **_) -> None:
        if not isinstance(pad_if_needed, bool):
            raise TypeError(f"pad_if_needed must be bool, got {type(pad_if_needed)}")

        if padding is not None:
            if not isinstance(padding, (int, list, tuple)):
                raise TypeError(
                    f"Padding must be an int or a sequence of length 1, 2 or 4, "
                    f"got {type(padding)}"
                )
            if isinstance(padding, (list, tuple)) and len(padding) not in (1, 2, 4):
                raise ValueError(f"Padding sequence must have length 1, 2 or 4, got {len(padding)}")
            if isinstance(padding, (list, tuple)) and any(
                not isinstance(value, int) for value in padding
            ):
                raise ValueError(f"Padding values must be integers, got {padding}")
            _ValidateIfNonNegative.verify(values=padding, name="padding")

        if pad_if_needed or padding is not None:
            _ValidatePaddingMode.verify(padding_mode=padding_mode)


class _ValidateFill(_ArgumentValidateRule):
    """
    Verify RandomCrop fill argument.
    """

    @classmethod
    def verify(cls, *, fill, **_) -> None:
        if fill is None or isinstance(fill, numbers.Number):
            return
        if isinstance(fill, (list, tuple)) and all(
            isinstance(value, numbers.Number) for value in fill
        ):
            if len(fill) == 0:
                raise ValueError("fill sequence must be non-empty")
            return
        raise TypeError(f"fill must be a number, sequence of numbers, or None, got {fill!r}")


class _ValidateRandomResizedCropScaleRatio(_ArgumentValidateRule):
    """
    Verify RandomResizedCrop scale and ratio arguments.
    """

    @classmethod
    def _verify_range(cls, value, name: str) -> None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"{name} should be a sequence of two numbers")
        if any(not isinstance(elem, numbers.Number) for elem in value):
            raise TypeError(f"{name} values must be numbers, got {value}")
        if any(elem <= 0 for elem in value):
            raise ValueError(f"{name} values must be positive, got {value}")
        if value[0] > value[1]:
            raise ValueError(f"{name} should be a (min, max) range, got {value}")

    @classmethod
    def verify(cls, *, scale, ratio, **_) -> None:
        cls._verify_range(scale, "scale")
        cls._verify_range(ratio, "ratio")


class _ValidateRandomResizedCropInterpolation(_ArgumentValidateRule):
    """
    Verify RandomResizedCrop interpolation argument.
    """

    @classmethod
    def verify(cls, *, interpolation, **_) -> None:
        Resize.validate_interpoliation(interpolation)


class RandomCrop(Operator):
    """
    Crop the input at a random location.

    If the input is a ``torch.Tensor`` it can have an arbitrary number of leading batch dimensions.
    For example, the image tensor can have [..., C, H, W] shape.

    Parameters
    ----------
    size : sequence or int
        Desired output size of the crop. If size is an int instead of sequence like (h, w),
        a square crop (size, size) is made. If provided a sequence of length 1, it will be
        interpreted as (size[0], size[0]).
    padding : int or sequence, optional, default = None
        Optional padding on each border of the image, applied before cropping. If a single int
        or a sequence of length 1 is provided this is used to pad all borders. If sequence of
        length 2 is provided this is the padding on left/right and top/bottom respectively. If
        a sequence of length 4 is provided this is the padding for the left, top, right and
        bottom borders respectively.
    pad_if_needed : bool, optional, default = False
        Pad the image if it is smaller than the desired size.
    fill : number or tuple, optional, default = 0
        Pixel fill value used when the padding_mode is constant.
    padding_mode : Literal["constant", "edge", "reflect", "symmetric"], optional, default="constant"
        Type of padding. Should be: constant, edge, reflect or symmetric.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the crop. Can be ``"cpu"`` or ``"gpu"``.
    """

    arg_rules = [_ValidateSizeDescriptor, _ValidateCropSize, _ValidatePadding, _ValidateFill]
    preprocess_data = get_HWC_from_layout_pipeline

    @classmethod
    def adjust_size(cls, size: int | Sequence[int]) -> Sequence[int]:
        return CenterCrop.adjust_size(size)

    @classmethod
    def adjust_padding(cls, padding: None | int | Sequence[int]) -> tuple[int, int, int, int]:
        if padding is None:
            return 0, 0, 0, 0
        if isinstance(padding, int):
            return padding, padding, padding, padding
        if isinstance(padding, (list, tuple)):
            if len(padding) == 1:
                return padding[0], padding[0], padding[0], padding[0]
            if len(padding) == 2:
                return padding[0], padding[1], padding[0], padding[1]
            if len(padding) == 4:
                return tuple(padding)

        raise TypeError(
            f"Padding must be an int or a sequence of length 1, 2 or 4, got {type(padding)}"
        )

    @staticmethod
    def adjust_fill(fill):
        if fill is None:
            return 0
        if isinstance(fill, numbers.Number):
            return fill
        return tuple(fill)

    @staticmethod
    def _randint(max_value):
        # Clamp to 0 so a negative max_value (crop larger than the available range)
        # collapses to a single valid position 0 instead of giving fn.random.uniform
        # an empty / inverted range.
        max_value = dali.math.max(max_value, 0)
        range_start = fn.cast(0, dtype=dali.types.FLOAT)
        range_end = fn.cast(max_value + 1, dtype=dali.types.FLOAT)
        value = dali.math.floor(fn.random.uniform(range=fn.stack(range_start, range_end)))
        return fn.cast(value, dtype=dali.types.INT32)

    def __init__(
        self,
        size: int | Sequence[int],
        padding: None | int | Sequence[int] = None,
        pad_if_needed: bool = False,
        fill: int | float | Sequence[int] | Sequence[float] | None = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            device=device,
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            padding_mode=padding_mode,
            fill=fill,
        )

        self.size = RandomCrop.adjust_size(size)
        self.padding = RandomCrop.adjust_padding(padding)
        self.pad_if_needed = pad_if_needed
        self.fill = RandomCrop.adjust_fill(fill)
        self.padding_mode = padding_mode
        self.needs_padding = pad_if_needed or any(self.padding)

    def _kernel(self, data_input):
        """
        Applies the random crop to the input data.
        """
        in_h, in_w, _, tensor = data_input
        crop_h, crop_w = self.size
        pad_left, pad_top, pad_right, pad_bottom = self.padding

        padded_h = in_h + pad_top + pad_bottom
        padded_w = in_w + pad_left + pad_right

        if self.pad_if_needed:
            pad_h = dali.math.max(crop_h - padded_h, 0)
            pad_w = dali.math.max(crop_w - padded_w, 0)
            pad_top = pad_top + pad_h
            pad_left = pad_left + pad_w
            # Only pad_top / pad_left are read below; pad_bottom / pad_right are dropped.
            padded_h = padded_h + 2 * pad_h
            padded_w = padded_w + 2 * pad_w

        max_top = fn.cast(padded_h, dtype=dali.types.INT32) - crop_h
        max_left = fn.cast(padded_w, dtype=dali.types.INT32) - crop_w

        top = RandomCrop._randint(max_top)
        left = RandomCrop._randint(max_left)

        slice_kwargs = {
            "device": self.device,
            "axis_names": "WH",
        }
        if self.needs_padding:
            slice_kwargs.update(
                out_of_bounds_policy=PADDING_CLASS[self.padding_mode].border_type,
                fill_values=self.fill,
            )

        return fn.slice(
            tensor,
            fn.stack(
                fn.cast(left - pad_left, dtype=dali.types.INT32),
                fn.cast(top - pad_top, dtype=dali.types.INT32),
            ),
            fn.stack(crop_w, crop_h),
            **slice_kwargs,
        )


class RandomResizedCrop(Operator):
    """
    Crop a random portion of the input and resize it to a given size.

    If the input is a ``torch.Tensor`` it can have an arbitrary number of leading batch dimensions.
    For example, the image tensor can have [..., C, H, W] shape.

    Parameters
    ----------
    size : sequence or int
        Expected output size of the crop. If size is an int instead of sequence like (h, w),
        a square output size (size, size) is made. If provided a sequence of length 1, it will be
        interpreted as (size[0], size[0]).
    scale : tuple of float, optional, default = (0.08, 1.0)
        Lower and upper bounds for the random crop area, relative to the input image area.
    ratio : tuple of float, optional, default = (3 / 4, 4 / 3)
        Lower and upper bounds for the random crop aspect ratio, width / height.
    interpolation : InterpolationMode or int, optional, default = InterpolationMode.BILINEAR
        Interpolation mode to use for resizing. Legacy PIL integer codes
        (``0`` = NEAREST, ``1`` = LANCZOS, ``2`` = BILINEAR, ``3`` = BICUBIC,
        ``4`` = BOX, ``5`` = HAMMING) are accepted for torchvision compatibility.
    antialias : bool, optional, default = True
        Whether to apply antialiasing during resize.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the crop. Can be ``"cpu"`` or ``"gpu"``.
    """

    arg_rules = [
        _ValidateSizeDescriptor,
        _ValidateCropSize,
        _ValidateRandomResizedCropScaleRatio,
        _ValidateRandomResizedCropInterpolation,
    ]
    preprocess_data = get_HWC_from_layout_pipeline

    @classmethod
    def adjust_size(cls, size: int | Sequence[int]) -> Sequence[int]:
        return CenterCrop.adjust_size(size)

    def __init__(
        self,
        size: int | Sequence[int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
        antialias: bool | None = True,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        interpolation = Resize.normalize_interpolation(interpolation)

        super().__init__(
            device=device,
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )

        self.size = RandomResizedCrop.adjust_size(size)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.interpolation = Resize.interpolation_modes[interpolation]
        self.antialias = antialias

    def _kernel(self, data_input):
        """
        Applies random resized crop to the input data.
        """
        _, _, _, tensor = data_input

        return fn.random_resized_crop(
            tensor,
            device=self.device,
            size=self.size,
            random_area=self.scale,
            random_aspect_ratio=self.ratio,
            interp_type=self.interpolation,
            antialias=self.antialias,
            num_attempts=10,
        )
