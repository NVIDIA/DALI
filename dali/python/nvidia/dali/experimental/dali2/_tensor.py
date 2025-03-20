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
from ._eval_context import EvalContext as _EvalContext
from . import _eval_mode

class TensorMetadata:
    def __init__(self, shape: Tuple[int, ...], dtype: DType, layout: str):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout


def _volume(shape: Tuple[int, ...]) -> int:
    ret = 1
    for s in shape:
        ret *= s
    return ret


class Tensor:
    def __init__(
        self,
        data: Any,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        layout: Optional[str] = None,
    ):
        if layout is None:
            layout = ""
        elif not isinstance(layout, str):
            raise ValueError(f"Layout must be a string, got {type(layout)}")

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
        else:
            import numpy as np

            self._backend = TensorCPU(np.array(data), layout, False)

        if device is not None:
            self.device = device if isinstance(device, Device) else Device(device)
        else:
            self.device = Device("cpu")

        if isinstance(self._backend, TensorCPU) and device.device_type != "cpu":
            self.assign(self.to_device(device))

        self._metadata = TensorMetadata(self._backend.shape(), dtype, layout)

    def cpu(self) -> "Tensor":
        return self.to_device(Device("cpu"))

    def gpu(self, index: Optional[int] = None) -> "Tensor":
        return self.to_device(Device("gpu", index))

    def to_device(self, device: Device) -> "Tensor":
        if self.device == device:
            return self
        else:
            with device:
                raise NotImplementedError("Copying to a different device is not implemented yet")
                # return fn.copy(self, device=device.device_type)

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

    @property
    def dtype(self) -> DType:
        return self._metadata.dtype

    @property
    def layout(self) -> str:
        return self._metadata.layout

    @property
    def size(self) -> int:
        return _volume(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.bytes

    @property
    def itemsize(self) -> int:
        return self.dtype.bytes

    def item(self) -> Any:
        if self.size != 1:
            raise ValueError(f"Tensor has {self.size} elements, expected 1")
        import numpy as np

        with _EvalContext.get():
            return np.array(self.cpu().evaluate()._backend).item()

    def evaluate(self):
        if self._backend is None:
            assert self._expression is not None
            self._backend = self._expression.evaluate()._backend
        return self

    def __getitem__(self, ranges: Any) -> "TensorSlice":
        if not isinstance(ranges, tuple):
            ranges = (ranges,)

        if all(_is_full_slice(r) or r is Ellipsis for r in ranges):
            return self
        else:
            return TensorSlice(self, ranges)


def _is_int_value(tested: Any, reference: int) -> bool:
    return isinstance(tested, int) and tested == reference


def _is_full_slice(r: Any) -> bool:
    if isinstance(r, slice):
        return (
            (r.start is None or _is_int_value(r.start, 0))
            and (r.stop is None)
            and (r.step is None or _is_int_value(r.step, 1))
        )
    else:
        return False


def _is_index(r: Any) -> bool:
    return not isinstance(r, slice) and not r is Ellipsis


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def _scalar_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    elif isinstance(value, Tensor):
        return value.item()
    else:
        raise ValueError(f"Unsupported type: {type(value)}")


class TensorSlice:
    def __init__(self, tensor: Tensor, ranges: Tuple[Any, ...]):
        self._tensor = tensor
        self._ranges = ranges
        self._ndim_dropped = 0
        self._shape = None
        self._absolute_ranges = None
        if len(ranges) > tensor.ndim:
            raise ValueError(
                f"Number of ranges ({len(ranges)}) is greater than the number of dimensions of the tensor ({tensor.ndim})"
            )
        for r in ranges:
            if _is_index(r):
                self._ndim_dropped += 1

    @property
    def ndim(self) -> int:
        return self._tensor.ndim - self._ndim_dropped

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            shape = []
            if self._absolute_ranges is None:
                self._absolute_ranges = self._canonicalize_ranges(self._ranges, self._tensor.shape)
            for r in self._absolute_ranges:
                if isinstance(r, slice):
                    shape.append((r.stop + r.step - r.start - 1) // r.step)
            self._shape = tuple(shape)
        return self._shape

    @staticmethod
    def _canonicalize_ranges(ranges, in_shape) -> Tuple[int, ...]:
        d = 0
        abs_ranges = []
        for i, r in enumerate(ranges):
            if r is Ellipsis:
                to_skip = len(in_shape) - len(ranges) + 1
                for _ in range(to_skip):
                    abs_ranges.append(slice(0, in_shape[d], 1))
                    d += 1
                continue
            if isinstance(r, slice):
                step = _scalar_value(r.step) if r.step is not None else 1
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                extent = in_shape[d]
                start, stop = 0, extent
                if r.start is not None:
                    start = _scalar_value(r.start)
                    if start < 0:
                        start += extent
                    start = _clamp(start, 0, extent)
                if r.stop is not None:
                    stop = _scalar_value(r.stop)
                    if stop < 0:
                        stop += extent
                    stop = _clamp(stop, start, extent)
                abs_ranges.append(slice(start, stop, step))
            else:
                idx = _scalar_value(r)
                if idx < 0:
                    idx += in_shape[d]
                if idx < 0 or idx >= in_shape[d]:
                    raise IndexError(
                        f"Index {idx} is out of bounds for dimension {d} with size {in_shape[d]}"
                    )
                abs_ranges.append(idx)
            d += 1
        while d < len(in_shape):
            abs_ranges.append(slice(0, in_shape[d], 1))
            d += 1

        return tuple(abs_ranges)

    def __getitem__(self, ranges: Any) -> "TensorSlice":
        if not isinstance(ranges, tuple):
            ranges = (ranges,)

        if all(_is_full_slice(r) or r is Ellipsis for r in ranges):
            return self
        else:
            ranges = self._canonicalize_ranges(ranges, self.shape)
            abs_ranges = list(self._absolute_ranges)
            i = 0
            for d, r in enumerate(self._absolute_ranges):
                if isinstance(r, slice):
                    if isinstance(ranges[i], slice):
                        start = r.start + ranges[i].start
                        stop = r.start + ranges[i].stop
                        step = r.step * ranges[i].step
                        abs_ranges[d] = slice(start, stop, step)
                    else:
                        abs_ranges[d] = r.start + ranges[i] * r.step
                    i += 1
            slice = TensorSlice(self._tensor, tuple(abs_ranges))
            if _eval_mode.eval_mode >= _eval_mode.EvalMode.EAGER:
                return slice.evaluate()
            else:
                return slice

    def evaluate(self):
        if not isinstance(ranges, (tuple)):
            ranges = (ranges,)

        with _EvalContext.get() as context:
            if len(ranges) == 0:
                return self._tensor.evaluate()

            if all(_is_full_slice(r) for r in ranges):
                return self._tensor.evaluate()

            args = {}
            d = 0
            for i, r in enumerate(ranges):
                if r is Ellipsis:
                    d += self.ndim - len(ranges)
                elif isinstance(r, slice):
                    if r.start is not None:
                        args[f"lo_{d}"] = r.start
                    if r.stop is not None:
                        args[f"hi_{d}"] = r.stop
                    if r.step is not None:
                        args[f"step_{d}"] = r.step
                    d += 1
                else:
                    args[f"at_{d}"] = r
                    d += 1

            return fn.tensor_subscript(self, *args).evaluate()
