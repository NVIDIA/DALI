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
from torch import Tensor
import nvidia.dali.experimental.dynamic as ndd
from torchvision.transforms import InterpolationMode

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
    """
    Please refer to the ``Resize`` operator for more details.
    """
    Resize.verify_args(
        size=size, max_size=max_size, interpolation=interpolation, antialias=antialias
    )

    effective_size, mode = Resize.infer_effective_size(size, max_size)
    interpolation = Resize.interpolation_modes[interpolation]

    if isinstance(img, ndd.Tensor):
        img_shape = img.shape
    elif isinstance(img, ndd.Batch):
        img_shape = img.shape[0]  # Batches have uniform layout
    else:
        raise TypeError(f"Input must be ndd.Tensor or ndd.Batch got {type(img)}")

    if img.layout in ["HWC", "NHWC"]:
        original_h = img_shape[-3]
        original_w = img_shape[-2]
    elif img.layout in ["CHW", "NCHW"]:
        original_h = img_shape[-2]
        original_w = img_shape[-1]

    target_h, target_w = Resize.calculate_target_size(
        (original_h, original_w), effective_size, max_size, size is None
    )

    # Shorter edge limited by max size
    if mode == "resize_shorter":
        return ndd.resize(
            img,
            device=device,
            resize_shorter=target_h,
            max_size=max_size,
            interp_type=interpolation,
            antialias=antialias,
        )

    return ndd.resize(
        img,
        device=device,
        size=(target_h, target_w),
        mode=mode,
        interp_type=interpolation,
        antialias=antialias,
    )
