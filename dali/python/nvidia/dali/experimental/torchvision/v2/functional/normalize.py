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

from typing import List
import nvidia.dali.experimental.dynamic as ndd

from nvidia.dali._typing import TensorLike
from nvidia.dali.experimental.dynamic._device import DeviceLike

from ..operator import adjust_input, _ValidateIsTensor
from ..normalize import Normalize, _channel_params


@adjust_input
def _normalize(
    input_data: TensorLike | ndd.Batch,
    mean: List[float],
    std: List[float],
    inplace: bool = False,
    device: DeviceLike = "cpu",
) -> TensorLike | ndd.Batch:

    # Torchvision does not support normalization on HWC, but DALI does
    mean, std = _channel_params(mean, std, channels_last=input_data.layout[-1] == "C")

    # adjust_input already moved the input to the requested device; the constants must follow it
    data_device = input_data.device.device_type
    mean = ndd.as_tensor(mean, device=data_device)
    std = ndd.as_tensor(std, device=data_device)
    return (input_data - mean) / std


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
