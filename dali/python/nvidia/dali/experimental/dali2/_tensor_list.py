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

from typing import Any, Optional, Tuple
from ._type import DType
from ._device import Device
from nvidia.dali.backend import TensorCPU, TensorGPU
from ._tensor import Tensor

class TensorList:
    def __init__(self, tensors: List[Any], dtype: Optional[DType] = None, device: Optional[Device] = None, layout: Optional[str] = None):
        self._tensors = []]
        if len(tensors) == 0:
            if dtype is None:
                raise ValueError("Element type must be specified if the list is empty")
            if device is None:
                device = Device("cpu")
            if layout is None:
                layout = ""
        else:
            for t in tensors:
                sample = Tensor(t, dtype, device, layout)
                if dtype is None:
                    dtype = sample.dtype
                if device is None:
                    device = sample.device
                if layout is None:
                    layout = sample.layout
                self._tensors.append(sample)

        self._dtype = dtype
        self._device = device
        self._layout = layout

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def tensors(self) -> List[Tensor]:
        return self._tensors



