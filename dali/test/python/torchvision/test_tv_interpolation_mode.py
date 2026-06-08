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

from enum import Enum

from nose_utils import assert_raises

from nvidia.dali.types import DALIInterpType
from nvidia.dali.experimental.torchvision import InterpolationMode, Resize
from nvidia.dali.experimental.torchvision.v2._enums import _normalize_enum_like_interpolation_mode


class TorchvisionLikeInterpolationMode(Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class InvalidInterpolationMode(Enum):
    BAD = "bad"


def test_interpolation_mode_is_exported_and_uses_torchvision_values():
    assert InterpolationMode.NEAREST.value == "nearest"
    assert InterpolationMode.NEAREST_EXACT.value == "nearest-exact"
    assert InterpolationMode.BILINEAR.value == "bilinear"
    assert InterpolationMode.BICUBIC.value == "bicubic"
    assert InterpolationMode.BOX.value == "box"
    assert InterpolationMode.HAMMING.value == "hamming"
    assert InterpolationMode.LANCZOS.value == "lanczos"


def test_interpolation_modes_match_torchvision():
    from torchvision.transforms import InterpolationMode as TorchvisionInterpolationMode

    dali_modes = {mode.name: mode.value for mode in InterpolationMode}
    torchvision_modes = {mode.name: mode.value for mode in TorchvisionInterpolationMode}

    assert dali_modes == torchvision_modes


def test_local_interpolation_mode_maps_to_dali_interpolation_type():
    assert Resize.normalize_interpolation(InterpolationMode.BILINEAR) is InterpolationMode.BILINEAR
    assert Resize.interpolation_modes[InterpolationMode.NEAREST] == DALIInterpType.INTERP_NN
    assert Resize.interpolation_modes[InterpolationMode.BILINEAR] == DALIInterpType.INTERP_LINEAR
    assert Resize.interpolation_modes[InterpolationMode.BICUBIC] == DALIInterpType.INTERP_CUBIC
    assert Resize.interpolation_modes[InterpolationMode.LANCZOS] == DALIInterpType.INTERP_LANCZOS3


def test_invalid_interpolation_errors_match_torchvision():
    from torchvision.transforms.v2.functional._geometry import _check_interpolation

    invalid_interpolations = ["bilinear", "bad", object(), InvalidInterpolationMode.BAD]

    for interpolation in invalid_interpolations:
        with assert_raises(ValueError) as torchvision_error:
            _check_interpolation(interpolation)

        with assert_raises(ValueError) as dali_error:
            _normalize_enum_like_interpolation_mode(interpolation)

        assert str(dali_error.exception) == str(torchvision_error.exception)


def test_enum_like_interpolation_mode_is_normalized_without_importing_torchvision():
    assert (
        Resize.normalize_interpolation(TorchvisionLikeInterpolationMode.BILINEAR)
        is InterpolationMode.BILINEAR
    )

    with assert_raises(NotImplementedError, glob="Interpolation mode"):
        Resize.validate_interpolation(TorchvisionLikeInterpolationMode.NEAREST_EXACT)


def test_invalid_interpolation_modes_still_raise():
    with assert_raises(ValueError, glob="Interpolation"):
        Resize.validate_interpolation("bilinear")

    with assert_raises(ValueError, glob="PIL code"):
        Resize.normalize_interpolation(99)
