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

from typing import Optional, List, Literal
import nvidia.dali.experimental.dynamic as ndd
from torchvision.transforms import InterpolationMode

import torch
from PIL import Image

from ..operator import adjust_input, get_HWC_from_layout_dynamic  # noqa: E402
from ..resize import Resize  # noqa: E402


@adjust_input
def resize(
    inpt: Image.Image | torch.Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = True,
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:
    """
    Please refer to the ``Resize`` operator for more details.
    """
    Resize.verify_args(
        size=size, max_size=max_size, interpolation=interpolation, antialias=antialias
    )

    size_normalized = Resize.infer_effective_size(size)
    interpolation = Resize.interpolation_modes[interpolation]

    original_h, original_w, _ = get_HWC_from_layout_dynamic(inpt)

    target_h, target_w = Resize.calculate_target_size_dynamic_mode(
        (original_h, original_w), size_normalized, max_size
    )

    return ndd.resize(
        inpt,
        device=device,
        size=(target_h, target_w),
        interp_type=interpolation,
        antialias=antialias,
    )
