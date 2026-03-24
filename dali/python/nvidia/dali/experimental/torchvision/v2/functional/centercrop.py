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

from typing import Literal, List
import nvidia.dali.experimental.dynamic as ndd
import torch
from PIL import Image
from ..operator import adjust_input
from ..centercrop import CenterCrop


@adjust_input
def center_crop(
    inpt: Image.Image | torch.Tensor,
    output_size: int | List[int],
    device: Literal["cpu", "gpu"] = "cpu",
) -> Image.Image | torch.Tensor:
    """
    Please refer to the ``CenterCrop`` operator for more details.
    """
    CenterCrop.verify_args(size=output_size)

    crop_h, crop_w = CenterCrop.adjust_size(output_size)

    # Derive H and W — mirrors the logic in functional/resize.py.
    if isinstance(inpt, ndd.Tensor):
        inpt_shape = inpt.shape
    elif isinstance(inpt, ndd.Batch):
        inpt_shape = inpt.shape[0]
    else:
        raise TypeError(f"Input must be ndd.Tensor or ndd.Batch, got {type(inpt)}")

    if inpt.layout[-3:] == "HWC":
        in_h, in_w = inpt_shape[-3], inpt_shape[-2]
    elif inpt.layout[-2:] in "HW":
        in_h, in_w = inpt_shape[-2], inpt_shape[-1]
    else:
        raise ValueError(
            f"Unsupported layout: {inpt.layout!r}. Expected one of HWC, NHWC, CHW, NCHW."
        )

    # torchvision: crop_top = int(round((H - crop_H) / 2.0))  — banker's rounding
    N_h, N_w = in_h - crop_h, in_w - crop_w
    crop_pos_y = int(round(N_h / 2.0)) / N_h if N_h > 0 else 0.5
    crop_pos_x = int(round(N_w / 2.0)) / N_w if N_w > 0 else 0.5

    return ndd.crop(
        inpt,
        device=device,
        crop=(crop_h, crop_w),
        crop_pos_x=crop_pos_x,
        crop_pos_y=crop_pos_y,
        out_of_bounds_policy="pad",
        fill_values=0,
    )
