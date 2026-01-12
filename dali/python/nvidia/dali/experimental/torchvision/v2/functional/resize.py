# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, List, Literal
from torch import Tensor
import nvidia.dali.experimental.dynamic as ndd
from torchvision.transforms import InterpolationMode

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..resize import Resize  # noqa: E402


@adjust_input
def resize(
    img: Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Tensor:

    Resize.verify_args(
        size=size, max_size=max_size, interpolation=interpolation, antialias=antialias
    )

    effective_size, mode = Resize.infer_effective_size(size, max_size)
    interpolation = Resize.interpolation_modes[interpolation]

    target_h, target_w = Resize.calculate_target_size(
        img.shape, effective_size, max_size, size is None
    )

    # Shorter edge limited by max size
    if mode == "resize_shorter":
        return ndd.resize(img, device=device, resize_shorter=target_h, max_size=max_size)

    return ndd.resize(
        img,
        device=device,
        size=(target_h, target_w),
        mode=mode,
    )
