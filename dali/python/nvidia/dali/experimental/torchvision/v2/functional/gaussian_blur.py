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

from typing import Literal, Sequence
from torch import Tensor
import nvidia.dali.experimental.dynamic as ndd

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..gaussian_blur import GaussianBlur  # noqa: E402


@adjust_input
def gaussian_blur(
    img: Tensor,
    kernel_size: int | Sequence[int],
    sigma: int | float | Sequence[float] = (0.1, 2.0),
    device: Literal["cpu", "gpu"] = "cpu",
) -> Tensor:

    GaussianBlur.verify_args(kernel_size=kernel_size, sigma=sigma)

    return ndd.gaussian_blur(img, window_size=kernel_size, sigma=sigma, device=device)
