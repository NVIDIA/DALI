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

import numpy as np
from typing import List, Literal
import torch
import nvidia.dali.experimental.dynamic as ndd

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402
from ..normalize import Normalize  # noqa: E402


@adjust_input
def _normalize(
    input_data: torch.Tensor,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
) -> ndd.Tensor:

    mean = np.asarray(mean)[:, None, None]
    std = np.asarray(std)[:, None, None]
    # TODO: support inplace

    return ndd.normalize(
        input_data,
        mean=mean,
        stddev=std,
        device=device,
    )


def normalize(
    input_data,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
) -> ndd.Tensor:
    """
    Please refer to the ``Normalize`` operator for more details.
    """
    Normalize.verify_args(std=std)
    Normalize.verify_data(input_data)

    return _normalize(
        input_data,
        mean=mean,
        std=std,
        device=device,
    )
