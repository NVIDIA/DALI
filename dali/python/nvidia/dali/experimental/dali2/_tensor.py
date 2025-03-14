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

class TensorMetadata:
    def __init__(self, shape: Tuple[int, ...], dtype: DType, layout: str):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

class Tensor:
    def __init__(self, data: Any, dtype: Optional[DType] = None, device: Optional[Device] = None, layout: Optional[str] = None):
        if isinstance(data, Tensor):
            if dtype is None or dtype == data.dtype:
                if device is None or device == data.device:
                    self.assign(data)
                else:
                    self.assign(data.to_device(device))
            else:
                self.assign(fn.cast(data, dtype, device=device))
            return
        elif hasattr(data, "__dlpack__"):
            self._backend = TensorCPU(data, layout)
        elif hasattr(data, "__array__"):
            self._backend = TensorCPU(data, layout)

        if device is not None:
            self.device = device if isinstance(device, Device) else Device(device)
        else:
            self.device = Device("cpu")

        self._metadata = TensorMetadata(self._backend.shape(), dtype, layout)

    def cpu(self) -> "Tensor":
        return self.to_device(Device("cpu"))

    def gpu(self) -> "Tensor":
        return self.to_device(Device("gpu"))

    def to_device(self, device: Device) -> "Tensor":
        if self.device == device:
            return self
        else:
            return fn.copy(self, device=device)

    def assign(self, other: "Tensor"):
        self._device = other._device
        self._metadata = other._metadata
        self._data = other._data
        self._backend = other._backend
        self._expression = other._expression

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._metadata.shape

    def __getitem__(self, ranges: Any) -> TensorSlice:
        if not isinstance(ranges, tuple):
            ranges = (ranges,)

        if all(_is_full_slice(r) or isinstance(r, Ellipsis) for r in ranges):
            return self
        else:
            return TensorSlice(self, ranges)

def _is_full_slice(r: Any) -> bool:
    if isinstance(r, slice):
        return (r.start is None or r.start == 0) and (r.stop is None) and (r.step is None or r.step == 1)
    else:
        return False

class TensorSlice:
    def __init__(self, tensor: Tensor, ranges: Tuple[Any, ...]):
        self._tensor = tensor
        self._ranges = ranges

