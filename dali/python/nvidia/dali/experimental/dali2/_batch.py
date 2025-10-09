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

from typing import Any, Optional, Union, Sequence
from ._type import DType, dtype as _dtype, type_id as _type_id
from ._tensor import (
    Tensor,
    _is_full_slice,
    _try_convert_enums,
    tensor as _tensor,
    as_tensor as _as_tensor,
)
import nvidia.dali.backend as _backend
from ._eval_context import EvalContext as _EvalContext
from ._device import Device
from . import _eval_mode
from . import _invocation
import nvtx


def _backend_device(backend: Union[_backend.TensorListCPU, _backend.TensorListGPU]) -> Device:
    if isinstance(backend, _backend.TensorListCPU):
        return Device("cpu")
    elif isinstance(backend, _backend.TensorListGPU):
        return Device("gpu", backend.device_id())
    else:
        raise ValueError(f"Unsupported backend type: {type(backend)}")


def _is_tensor_type(x):
    from . import _batch

    if isinstance(x, _batch.Batch):
        raise ValueError("A list of Batch objects is not a valid argument type")
    if isinstance(x, Tensor):
        return True
    if hasattr(x, "__array__"):
        return True
    if hasattr(x, "__cuda_array_interface__"):
        return True
    if hasattr(x, "__dlpack__"):
        return True
    return False


def _get_batch_size(x):
    if isinstance(x, Batch):
        return x.batch_size
    if isinstance(x, (_backend.TensorListCPU, _backend.TensorListGPU)):
        return len(x)
    return None


class BatchedSlice:
    def __init__(self, batch: "Batch"):
        self._batch = batch

    def __getitem__(self, ranges: Any) -> "Batch":
        if not isinstance(ranges, tuple):
            ranges = (ranges,)
        if len(ranges) == 0:
            return self._batch

        if all(_is_full_slice(r) for r in ranges):
            return self._batch

        args = {}
        d = 0
        for i, r in enumerate(ranges):
            if r is Ellipsis:
                d = self._batch.ndim - len(ranges) + i + 1
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

        from . import tensor_subscript

        return tensor_subscript(self._batch, **args)


def _arithm_op(name, *args, **kwargs):
    from . import arithmetic_generic_op

    argsstr = " ".join(f"&{i}" for i in range(len(args)))
    return arithmetic_generic_op(*args, expression_desc=f"{name}({argsstr})")


class _TensorList:
    def __init__(self, batch: "Batch", indices: Optional[Union[list[int], range]] = None):
        self._batch = batch
        self._indices = indices or range(batch.batch_size)

    def __getitem__(self, selection: Union[int, slice, list[int]]):
        return self.select(selection)

    def __len__(self):
        return len(self._indices)

    def select(self, selection):
        if selection == slice(None, None, None):
            return self
        if isinstance(selection, slice):
            return _TensorList(self._batch, self._indices[selection])
        elif isinstance(selection, list):
            return _TensorList(self._batch, [self._indices[i] for i in selection])
        else:
            return self._batch.select(selection)

    def tolist(self):
        return [self._batch._get_tensor(i) for i in self._indices]

    def as_batch(self):
        return as_batch(self)


class Batch:
    def __init__(
        self,
        tensors: Optional[Any] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        layout: Optional[str] = None,
        invocation_result: Optional[_invocation.InvocationResult] = None,
        copy: bool = False,
    ):
        assert isinstance(layout, str) or layout is None
        if device is not None and not isinstance(device, Device):
            device = Device(device)
        self._wraps_external_data = False
        self._tensors = None
        self._backend = None
        self._dtype = None
        self._device = None
        copied = False
        if tensors is not None:
            if isinstance(tensors, (_backend.TensorListCPU, _backend.TensorListGPU)):
                backend_dev = _backend_device(tensors)
                if (
                    (device is None or device == backend_dev)
                    and (dtype is None or dtype.type_id == tensors.dtype)
                    and (layout is None or layout == tensors.layout())
                ):
                    self._backend = tensors
                    self._device = backend_dev
                    self._layout = tensors.layout()
                    self._dtype = _dtype(tensors.dtype)
                else:
                    tmp = Batch(tensors)
                    if dtype is not None and dtype != tmp.dtype:
                        from . import cast

                        tmp = cast(tmp, dtype, device=device)
                        copied = True
                    # TODO(michalz): move before cast after we have proper cast op
                    if device is not None and device != tmp.device:
                        tmp = tmp.to_device(device)
                        copied = True
                    self.assign(tmp)
                    if self._backend and layout:
                        self._backend.set_layout(layout)
            elif _is_tensor_type(tensors):
                if copy:
                    t = _tensor(tensors, dtype=dtype, device=device, layout=layout)
                else:
                    t = _as_tensor(tensors, dtype=dtype, device=device, layout=layout)
                if t.ndim == 0:
                    raise ValueError("Cannot create a batch from a scalar")
                if dtype is None:
                    dtype = t.dtype
                if device is None:
                    device = t.device
                if layout is None:
                    layout = t.layout
                if t._backend is not None:
                    if isinstance(t._backend, _backend.TensorCPU):
                        self._backend = _backend.TensorListCPU(t._backend, layout=layout)
                    elif isinstance(t._backend, _backend.TensorGPU):
                        self._backend = _backend.TensorListGPU(t._backend, layout=layout)
                    else:
                        raise ValueError(f"Unsupported device type: {t.device.device_type}")
                    if t._wraps_external_data:
                        self._wraps_external_data = True
                else:
                    sh = t.shape
                    tensors = [t[i] for i in range(sh[0])]
                self._dtype = dtype

            else:
                self._tensors = []
                for i, t in enumerate(tensors):
                    if t is None:
                        raise TypeError(
                            f"Tensors must be array-like types or numbers. Got `None` at index {i}"
                        )
                    sample = Tensor(t, dtype=dtype, device=device, layout=layout)
                    if dtype is None:
                        dtype = sample.dtype
                    if device is None:
                        device = sample.device
                    if layout is None:
                        layout = sample.layout
                    self._tensors.append(sample)
                    if sample._wraps_external_data:
                        self._wraps_external_data = True
                    else:
                        if not isinstance(t, Tensor) or t._backend is not sample._backend:
                            copied = True
                if dtype is None:
                    # We would have set dtype in the 1st iteration, so the only way it can
                    # be None is if the `_tensors` are empty.
                    assert len(self._tensors) == 0
                    raise ValueError("Element type must be specified if the list is empty")
                if device is None:
                    device = Device("cpu")
                if layout is None:
                    layout = ""
                self._device = device
                self._layout = layout
                self._dtype = dtype
                if len(self._tensors) == 0:
                    t = Tensor([], dtype=dtype, device=device).evaluate()
                    if self._device.device_type == "cpu":
                        backend_type = _backend.TensorListCPU
                    elif self._device.device_type == "gpu":
                        backend_type = _backend.TensorListGPU
                    self._backend = backend_type(t._backend, layout=layout)

        if self._dtype is None:
            if self._backend is not None:
                self._dtype = DType.from_type_id(self._backend.dtype)
            else:
                self._dtype = dtype
        if self._device is None:
            if self._backend is not None:
                self._device = _backend_device(self._backend)
            else:
                self._device = device
        self._layout = layout
        self._invocation_result = invocation_result
        self._ndim = None
        if self._tensors and self._tensors[0]._shape:
            self._ndim = len(self._tensors[0]._shape)

        if copy and not copied:
            dev = self.to_device(self.device, force_copy=True)
            if dtype is not None and dev.dtype != dtype:
                from . import cast

                dev = cast(dev, dtype, device=device)
            self.assign(dev.evaluate())
            copied = True
        else:
            if self._dtype is not None and dtype is not None and self._dtype != dtype:
                from . import cast

                self.assign(cast(self, dtype, device=device))

        if _eval_mode.EvalMode.current().value >= _eval_mode.EvalMode.eager.value:
            self.evaluate()

    def _is_external(self) -> bool:
        return self._wraps_external_data

    @staticmethod
    def broadcast(sample, batch_size: int, device: Optional[Device] = None) -> "Batch":
        if isinstance(sample, Batch):
            raise ValueError("Cannot broadcast a Batch")
        if _is_tensor_type(sample):
            t = _as_tensor(sample, device=device).evaluate()
            if t.device.device_type == "gpu":
                tl_type = _backend.TensorListGPU
            else:
                tl_type = _backend.TensorListCPU
            return Batch(tl_type.broadcast(t._backend, batch_size))
        import numpy as np

        with nvtx.annotate("to numpy and stack", domain="batch"):
            arr = np.array(sample)
            converted_dtype_id = None
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int64:
                arr = arr.astype(np.int32)
            elif arr.dtype == np.uint64:
                arr = arr.astype(np.uint32)
            elif arr.dtype == object:
                arr, converted_dtype_id = _try_convert_enums(arr)
            arr = np.repeat(arr[np.newaxis], batch_size, axis=0)

        with nvtx.annotate("to backend", domain="batch"):
            tl = _backend.TensorListCPU(arr)
            if converted_dtype_id is not None:
                tl.reinterpret(converted_dtype_id)
        with nvtx.annotate("create batch", domain="batch"):
            return Batch(tl, device=device)

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
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        return self._dtype

    @property
    def device(self) -> Device:
        if self._device is None:
            if self._invocation_result is not None:
                self._device = self._invocation_result.device
            elif self._tensors:
                self._device = self._tensors[0].device
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        return self._device

    @property
    def layout(self) -> str:
        if self._layout is None:
            if self._invocation_result is not None:
                self._layout = self._invocation_result.layout
            elif self._backend is not None:
                self._layout = self._backend.layout()
                if self._layout == "" and self._ndim != 0:
                    self._layout = None
            elif self._tensors:
                self._layout = self._tensors[0].layout
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        # Use "" to indicate that the layout has been checked and is empty, but still return None
        # to avoid situations where we return a string with a length that doesn't match the number
        # of dimensions.
        return self._layout or None

    @property
    def ndim(self) -> int:
        if self._ndim is None:
            if self._invocation_result is not None:
                self._ndim = self._invocation_result.ndim
            elif self._backend is not None:
                self._ndim = self._backend.ndim()
            elif self._tensors:
                self._ndim = self._tensors[0].ndim
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        return self._ndim

    @property
    def tensors(self):
        return _TensorList(self)

    def to_device(self, device: Device, force_copy: bool = False) -> "Batch":
        if device is not None and not isinstance(device, Device):
            device = Device(device)
        if self.device == device and not force_copy:
            return self
        else:
            with device:
                from . import copy

                ret = copy(self, device=device)
                return ret

    def cpu(self) -> "Batch":
        return self.to_device(Device("cpu"))

    def gpu(self, index: Optional[int] = None) -> "Batch":
        return self.to_device(Device("gpu", index))

    def assign(self, other: "Batch"):
        if other is self:
            return
        self._device = other._device
        self._dtype = other._dtype
        self._layout = other._layout
        self._backend = other._backend
        self._tensors = other._tensors
        self._invocation_result = other._invocation_result
        self._wraps_external_data = other._wraps_external_data

    @property
    def slice(self):
        """Interface for samplewise slicing.

        Regular slicing selects samples first and then slices each sample with common
        slicing parameters.

        Samplewise slicing interface allows the slicing parmaters to be batches (with the same
        number of samples) and the slicing parameters are applied to respective samples.

        ```Python
        start = Batch([1, 2, 3])
        stop = Batch([4, 5, 6])
        step = Batch([1, 1, 2])
        sliced = input[start, stop, step]
        # the result is equivalent to
        sliced = Batch([
            sample[start[i]:stop[i]:step[i]]
            for i, sample in enumerate(input)
        ])
        ```

        If the slicing parameters are not batches, they are broadcast to all samples.
        """
        return BatchedSlice(self)

    def __iter__(self):
        return iter(self.tensors)

    def select(self, r):
        if r is ...:
            return self
        if isinstance(r, slice):
            return Batch(self.tensors[r])
        elif isinstance(r, list):
            return Batch(self.tensors[r])
        else:
            return self._get_tensor(r)

    def _get_tensor(self, i):
        if self._tensors is None:
            self._tensors = [None] * self.batch_size

        t = self._tensors[i]
        if t is None:
            t = self._tensors[i] = Tensor(batch=self, index_in_batch=i)
            if self._backend:
                t._backend = self._backend[i]
        return t

    def _plain_slice(self, ranges):
        def _is_batch(x):
            return _get_batch_size(x) is not None

        for r in ranges:
            is_batch_arg = _is_batch(r)
            if isinstance(r, slice):
                if _is_batch(r.start) or _is_batch(r.stop) or _is_batch(r.step):
                    is_batch_arg = True
            if is_batch_arg:
                raise ValueError(
                    "Cannot use a batch as an index or slice. in ``Batch.__getitem__``.\n"
                    "Use ``.slice`` property to perform samplewise slicing."
                )
        return self.slice.__getitem__(ranges)

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

    @property
    def shape(self):
        if self._invocation_result is not None:
            return self._invocation_result.shape
        if self._backend is not None:
            return self._backend.shape()
        else:
            assert self._tensors is not None
            return [t.shape for t in self._tensors]

    def __str__(self) -> str:
        return "Batch(\n" + str(self.evaluate()._backend) + ")"

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

    def __add__(self, other):
        return _arithm_op("add", self, other)

    def __radd__(self, other):
        return _arithm_op("add", other, self)

    def __sub__(self, other):
        return _arithm_op("sub", self, other)

    def __rsub__(self, other):
        return _arithm_op("sub", other, self)

    def __mul__(self, other):
        return _arithm_op("mul", self, other)

    def __rmul__(self, other):
        return _arithm_op("mul", other, self)

    def __pow__(self, other):
        return _arithm_op("pow", self, other)

    def __rpow__(self, other):
        return _arithm_op("pow", other, self)

    def __truediv__(self, other):
        return _arithm_op("fdiv", self, other)

    def __rtruediv__(self, other):
        return _arithm_op("fdiv", other, self)

    def __floordiv__(self, other):
        return _arithm_op("div", self, other)

    def __rfloordiv__(self, other):
        return _arithm_op("div", other, self)

    def __neg__(self):
        return _arithm_op("minus", self)

    # Short-circuiting the execution, unary + is basically a no-op
    def __pos__(self):
        return self

    def __eq__(self, other):
        return _arithm_op("eq", self, other)

    def __ne__(self, other):
        return _arithm_op("neq", self, other)

    def __lt__(self, other):
        return _arithm_op("lt", self, other)

    def __le__(self, other):
        return _arithm_op("leq", self, other)

    def __gt__(self, other):
        return _arithm_op("gt", self, other)

    def __ge__(self, other):
        return _arithm_op("geq", self, other)

    def __and__(self, other):
        return _arithm_op("bitand", self, other)

    def __rand__(self, other):
        return _arithm_op("bitand", other, self)

    def __or__(self, other):
        return _arithm_op("bitor", self, other)

    def __ror__(self, other):
        return _arithm_op("bitor", other, self)

    def __xor__(self, other):
        return _arithm_op("bitxor", self, other)

    def __rxor__(self, other):
        return _arithm_op("bitxor", other, self)


def batch(
    tensors: Union[Batch, Sequence[Any]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    layout: Optional[str] = None,
):
    if isinstance(tensors, Batch):
        b = tensors.to_device(device or tensors.device, force_copy=True)
        if dtype is not None and b.dtype != dtype:
            from . import cast

            b = cast(b, dtype, device=device)
        return b.evaluate()
    else:
        return Batch(tensors, dtype=dtype, device=device, layout=layout, copy=True)


def as_batch(
    tensors: Union[Batch, Sequence[Any]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    layout: Optional[str] = None,
):
    if isinstance(tensors, Batch):
        b = tensors
        if device is not None:
            b = tensors.to_device(device)
        if dtype is not None and b.dtype != dtype:
            from . import cast

            b = cast(b, dtype, device=device)
        return b
    else:
        return Batch(tensors, dtype=dtype, device=device, layout=layout)


__all__ = ["Batch", "batch", "as_batch"]
