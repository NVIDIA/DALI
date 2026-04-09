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

import numpy as np
from typing import List
import nvidia.dali.experimental.dynamic as ndd

from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input, _ValidateIsTensor
from ..normalize import Normalize


@adjust_input
def _normalize(
    input_data: TensorLike | ndd.Batch,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
    device: DeviceLike = "cpu",
) -> TensorLike | ndd.Batch:

    if input_data.layout[-3:] == "CHW":
        mean = np.asarray(mean)[:, None, None]
        std = np.asarray(std)[:, None, None]
    else:
        # Torchvison does not support nomrlization on HWC, but DALI does
        mean = np.asarray(mean)[None, None, :]
        std = np.asarray(std)[None, None, :]

    return ndd.normalize(
        input_data,
        mean=mean,
        stddev=std,
        device=device,
    )


def normalize(
    input_data: TensorLike | ndd.Batch,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
    device: DeviceLike = "cpu",
) -> TensorLike | ndd.Batch:
    """
    Please refer to the ``Normalize`` operator for more details.
    """
    Normalize.verify_args(std=std, mean=mean)
    _ValidateIsTensor.verify(input_data)

    if inplace:
        raise NotImplementedError("inplace is not implemented, yet")

    return _normalize(
        input_data,
        mean=mean,
        std=std,
        device=device,
    )
