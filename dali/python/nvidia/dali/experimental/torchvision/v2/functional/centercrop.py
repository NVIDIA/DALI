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
from torch import Tensor
import nvidia.dali.experimental.dynamic as ndd

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..centercrop import CenterCrop  # noqa: E402


@adjust_input
def center_crop(
    img: Tensor,
    output_size: List[int],
    device: Literal["cpu", "gpu"] = "cpu",
) -> Tensor:
    """
    Please refer to the ``CenterCrop`` operator for more details.
    """
    CenterCrop.verify_args(size=output_size)

    return ndd.crop(
        img,
        device=device,
        crop=CenterCrop.adjust_size(output_size),
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        out_of_bounds_policy="pad",
        fill_values=0,
    )
