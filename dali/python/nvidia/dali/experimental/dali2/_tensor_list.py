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

from typing import Any, Optional, Tuple, List, Union
from ._type import DType, dtype as _dtype
from ._tensor import Tensor, _is_full_slice
import nvidia.dali.backend as _backend
from ._eval_context import EvalContext as _EvalContext
from ._device import Device
from . import _invocation

class _TensorListRange:
    def __init__(self, backend: Any, start: int = 0, stop: int = -1, step: int = 1):
        if self._step == 0:
            raise ValueError("Step cannot be 0")
        self._backend = backend
        n = len(backend)
        if start < 0:
            start += n
        if stop < 0:
            stop += n
        start = min(max(start, 0), n)
        stop = min(max(stop, 0), n)
        self._start = start
        self._stop = stop
        self._step = step
        if step > 0:
            self._stop = max(self._start, self._stop)
        else:
            self._stop = min(self._start, self._stop)

    def __len__(self) -> int:
        return (self._stop - self._start) // self._step

    def __getitem__(self, range: Any) -> Union["_TensorListRange", "Tensor"]:
        if isinstance(range, tuple):
            raise ValueError(
                "A TensorList is a 1D object. Use a single index instead or a single slice."
            )
        if isinstance(range, slice):
            start = self._start
            stop = self._stop
            step = self._step
            if step == 0:
                raise ValueError("Step cannot be 0")
            if range.start is not None:
                start = range.start + self._start
                if range.start < 0:
                    start += len(self)
            if range.stop is not None:
                stop = range.stop + self._start
                if range.stop < 0:
                    stop += len(self)
            start = min(max(start, self._start), self._stop)
            stop = min(max(stop, start), self._stop)
            if range.step is not None:
                step *= range.step
            return _TensorListRange(self._backend, start, stop, step)
        else:
            if range < 0:
                range += len(self)
            if range < 0 or range >= len(self):
                raise IndexError(
                    f"The index {range} is out of bounds for the TensorList of size {len(self)}"
                )

class SamplewiseOp:
    def __init__(self, tensor_list: "TensorList"):
        self._tensor_list = tensor_list

    def __getitem__(self, ranges: Any) -> "TensorList":
        if not isinstance(ranges, (tuple)):
            ranges = (ranges,)
        if len(ranges) == 0:
            return self._tensor_list

        if all(_is_full_slice(r) for r in ranges):
            return self._tensor_list

        args = {}
        d = 0
        for i, r in enumerate(ranges):
            if r is Ellipsis:
                d = self.ndim - len(ranges) + i + 1
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

        from . import fn
        return fn.tensor_subscript(self._tensor_list, **args)

    def __add__(self, other):
        return _arithm_op("add", self._tensor_list, other)

    def __radd__(self, other):
        return _arithm_op("add", other, self._tensor_list)

    def __sub__(self, other):
        return _arithm_op("sub", self._tensor_list, other)

    def __rsub__(self, other):
        return _arithm_op("sub", other, self._tensor_list)

    def __mul__(self, other):
        return _arithm_op("mul", self._tensor_list, other)

    def __rmul__(self, other):
        return _arithm_op("mul", other, self._tensor_list)

    def __pow__(self, other):
        return _arithm_op("pow", self._tensor_list, other)

    def __rpow__(self, other):
        return _arithm_op("pow", other, self._tensor_list)

    def __truediv__(self, other):
        return _arithm_op("fdiv", self._tensor_list, other)

    def __rtruediv__(self, other):
        return _arithm_op("fdiv", other, self._tensor_list)

    def __floordiv__(self, other):
        return _arithm_op("div", self._tensor_list, other)

    def __rfloordiv__(self, other):
        return _arithm_op("div", other, self._tensor_list)

    def __neg__(self):
        return _arithm_op("minus", self._tensor_list)

    # Short-circuiting the execution, unary + is basically a no-op
    def __pos__(self):
        return self

    def __eq__(self, other):
        return _arithm_op("eq", self._tensor_list, other)

    def __ne__(self, other):
        return _arithm_op("neq", self._tensor_list, other)

    def __lt__(self, other):
        return _arithm_op("lt", self._tensor_list, other)

    def __le__(self, other):
        return _arithm_op("leq", self._tensor_list, other)

    def __gt__(self, other):
        return _arithm_op("gt", self._tensor_list, other)

    def __ge__(self, other):
        return _arithm_op("geq", self._tensor_list, other)

    def __and__(self, other):
        return _arithm_op("bitand", self._tensor_list, other)

    def __rand__(self, other):
        return _arithm_op("bitand", other, self._tensor_list)

    def __or__(self, other):
        return _arithm_op("bitor", self._tensor_list, other)

    def __ror__(self, other):
        return _arithm_op("bitor", other, self._tensor_list)

    def __xor__(self, other):
        return _arithm_op("bitxor", self._tensor_list, other)

    def __rxor__(self, other):
        return _arithm_op("bitxor", other, self._tensor_list)

def _arithm_op(name, *args, **kwargs):
    from . import fn

    argsstr = " ".join(f"&{i}" for i in range(len(args)))
    return fn.arithmetic_generic_op(*args, expression_desc=f"{name}({argsstr})")


class TensorList:
    def __init__(
        self,
        tensors: Optional[List[Any]] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        layout: Optional[str] = None,
        invocation_result: Optional[_invocation.InvocationResult] = None,
    ):
        assert(isinstance(layout, str) or layout is None)
        self._tensors = None
        if tensors is not None:
            self._tensors = []
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

        if dtype is not None:
            if not isinstance(dtype, DType):
                dtype = _dtype(dtype)
        self._dtype = dtype
        self._device = device
        self._layout = layout
        self._backend = None
        self._invocation_result = invocation_result
        self._ndim = None
        if self._tensors and self._tensors[0]._shape:
            self._ndim = len(self._tensors[0]._shape)

    @property
    def dtype(self) -> DType:
        if self._dtype is None:
            if self._backend is not None:
                self._dtype = DType.from_type_id(self._backend.dtype)
            elif self._invocation_result is not None:
                self._dtype = _dtype(self._invocation_result.dtype)
            elif self._tensors:
                self._dtype = self._tensors[0].dtype
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty TensorList")
        return self._dtype

    @property
    def device(self) -> Device:
        if self._device is None:
            if self._invocation_result is not None:
                self._device = self._invocation_result.device
                print("From invocation result", self._device)
            elif self._tensors:
                self._device = self._tensors[0].device
                print("From tensors", self._device)
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty TensorList")
        return self._device

    @property
    def layout(self) -> str:
        if self._layout is None:
            if self._invocation_result is not None:
                self._layout = self._invocation_result.layout
            elif self._tensors:
                self._layout = self._tensors[0].layout
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty TensorList")
        return self._layout

    @property
    def ndim(self) -> int:
        if self._ndim is None:
            if self._invocation_result is not None:
                self._ndim = self._invocation_result.ndim
            elif self._tensors:
                self._ndim = self._tensors[0].ndim
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty TensorList")
        return self._ndim

    @property
    def tensors(self):
        if self._backend is not None:
            return _TensorListRange(self._backend)
        return self._tensors

    @property
    def batch_size(self) -> int:
        if self._backend is not None:
            return len(self._backend)
        elif self._tensors is not None:
            return len(self._tensors)
        elif self._invocation_result is not None:
            return self._invocation_result.batch_size
        else:
            raise ValueError("Neither tensors nor invocation result are set")

    def _is_same_tensor_list(self, other: "TensorList") -> bool:
        if self is other:
            return True
        return (
            self._backend is other._backend
            and self._invocation_result is other._invocation_result
            and (
                self._tensors is other._tensors
                or [t._is_same_tensor(ot) for t, ot in zip(self._tensors, other._tensors)]
            )
        )

    @property
    def shape(self):
        if self._invocation_result is not None:
            return self._invocation_result.shape
        return [t.shape for t in self._tensors]

    def __len__(self) -> int:
        return self.batch_siz

    @property
    def samplewise(self):
        return SamplewiseOp(self)

    def __str__(self) -> str:
        return "TensorList(\n" + str(self.evaluate()._backend) + ")"

    def evaluate(self):
        with _EvalContext.get() as ctx:
            if self._backend is None:
                if self._invocation_result is not None:
                    self._backend = self._invocation_result.value(ctx)
                else:
                    if self._device.device_type == "cpu":
                        backend_type = _backend.TensorListCPU
                    elif self._device.device_type == "gpu":
                        backend_type = _backend.TensorListGPU
                    else:
                        raise ValueError(
                            f"Internal error: Unsupported device type: {self._device.device_type}"
                        )
                    self._backend = backend_type(
                        [t.evaluate()._backend for t in self._tensors], self.layout
                    )
        return self
