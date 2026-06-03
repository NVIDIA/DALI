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


class InterpolationMode(Enum):
    """Interpolation modes compatible with ``torchvision.transforms.InterpolationMode``."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _normalize_enum_like_interpolation_mode(interpolation):
    """Normalize local and torchvision-like interpolation enums without importing torchvision."""
    if isinstance(interpolation, InterpolationMode):
        return interpolation

    name = getattr(interpolation, "name", None)
    if isinstance(name, str) and name in InterpolationMode.__members__:
        return InterpolationMode[name]

    value = getattr(interpolation, "value", None)
    if isinstance(value, str):
        try:
            return InterpolationMode(value)
        except ValueError:
            pass

    return interpolation


__all__ = ["InterpolationMode"]
