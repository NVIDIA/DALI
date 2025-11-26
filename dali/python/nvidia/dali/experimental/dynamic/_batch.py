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
from ._type import DType, dtype as _dtype
from ._tensor import (
    Tensor,
    _is_full_slice,
    _try_convert_enums,
    tensor as _tensor,
    as_tensor as _as_tensor,
)
import nvidia.dali.backend as _backend
import nvidia.dali.types as _dali_types
from ._device import Device, device as _device
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
    gpu = False
    new_args = [None] * len(args)
    for i, a in enumerate(args):
        if isinstance(a, (Batch, Tensor)):
            if a.device.device_type == "gpu":
                gpu = True
        else:
            # TODO(michalz): We might use some caching here for common values.
            if new_args is None:
                new_args = list(args)
            if gpu:
                new_args[i] = _as_tensor(a, device="gpu")
            else:
                new_args[i] = _as_tensor(a)
                if new_args[i].device.device_type == "gpu":
                    gpu = True

    for i in range(len(args)):
        if new_args[i] is None:
            if (args[i].device.device_type == "gpu") != gpu:
                raise ValueError("Cannot mix GPU and CPU inputs.")
            new_args[i] = args[i]

    return arithmetic_generic_op(*new_args, expression_desc=f"{name}({argsstr})")


class _TensorList:
    # `_TensorList` is what you get from `batch.tensors`.
    # `_TensorList` is private because it's never meant to be constructed by the user and merely
    # serves as an indexable proxy for tensor access. It's not a plain Python list because the
    # individual `Tensor` objects are created on demand - for example, if your Batch is a result
    # of running an operator, it will wrap a TensorListCPU/GPU. Accessing a single sample will
    # create just one Tensor object, not `batch_size` of them. One can imagine that at least some
    # users will use 0-th tensor to inspect some properties (instead of doing it on the arguably
    # less familiar batch level) and `_TensorList` will facilitate that without the overhead of
    # going over the entire batch. Returning a regular Python list would require us to eagerly
    # populate it - and even worse, we'd have to copy it each time, because otherwise a user
    # could try something like `batch.tensors.append(T)` which would make the list inconsistent.

    def __init__(self, batch: "Batch", indices: Optional[Union[list[int], range]] = None):
        self._batch = batch
        self._indices = indices or range(batch.batch_size)

    def __getitem__(self, selection: Union[int, slice, list[int]]):
        return self.select(selection)

    def __len__(self):
        return len(self._indices)

    def select(self, selection):
        """
        Selects a range of samples.

        The result of this function is either a :class:`_TensorList` (if `selection` is a `slice` or
        a `list`) or a :class:`Tensor` if `selection` is a number.
        """
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

    def as_batch(self, copy: bool = False):
        """
        Converts the list of tensors to a :class:`Batch` object.
        """
        return batch(self) if copy else as_batch(self)


class Batch:
    """A Batch object.

    This class represents a batch of tensors usable with DALI dynamic API. The tensors in the batch
    have the same element type, layout and number of dimensions, but can differ in shape.

    A :class:`Batch` can contain:

    * a single buffer and shape, owned by DALI, representing consecutive tensors
    * a list of :class:`Tensor` objects.
    * a result of a lazy evaluation of a DALI operator.

    In case of lazy evaluation, the operations are executed only after an attempt is made to access
    the tensor data or properties which cannot be obtained without running the underlying operation.
    """

    def __init__(
        self,
        tensors: Optional[Any] = None,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        layout: Optional[str] = None,
        invocation_result: Optional[_invocation.InvocationResult] = None,
        copy: bool = False,
    ):
        """Constructs a :class:`Batch` object.

        .. warning::
            :class:`Batch` objects should not be constructed directly, use :meth:`batch` or
            :meth:`as_batch` instead.

        The batch object can be created either from an existing object, passed as `tensors` or
        from an invocation result.
        Unless explicitly requested with the `copy` parameter, this constructor will make best
        effort to avoid the copy.

        Parameters
        ----------
        tensors : TensorLike, default: None
            The data to construct the batch from. It can be a list of tensors, a TensorList,
            or other supported types. If None, the batch is constructed from an invocation result.
            Supported types are:

            - a list of tensor-like objects; the objects need to have matching number of dimensions,
            data types and layouts,
            - a tensor-like object; the outermost dimenion is interpreted as the batch dimension
            - a dali.backend.TensorListCPU or dali.backend.TensorListGPU
        dtype : DType, default: None
            The desired data type of the batch. If not specified, the data type is inferred
            from the input tensors. If specified, the input tensors are cast to the desired
            data type. The `dtype` is required if `tensors` are an empty list.
        device : Device or str, optional, default: None
            The device on which the batch should reside (e.g., "cpu" or "gpu").
            If not specified, the device is inferred from the input tensors.
        layout : str, optional, default: None
            The layout string describing the dimensions of the batch (e.g., "HWC").
            If not specified, the layout is inferred from the input tensors.
        invocation_result : _invocation.InvocationResult, default: None
            The result of a DALI operator invocation, used for lazy evaluation
        copy : bool, optional, default: False
            If True, the input tensors are copied. If False, the constructor will avoid
            copying data when possible.
        """
        assert isinstance(layout, str) or layout is None
        if device is not None and not isinstance(device, Device):
            device = _device(device)
        self._wraps_external_data = False
        self._tensors = None  # The list of Tensor objects that comprise the batch.
        # This list is populated lazily when the batch contains a TensorList
        # or when it is a result of a batch operator invocation.
        self._storage = None  # The backing storage of the batch, TensorListCPU or TensorListGPU.
        self._dtype = None
        self._device = None
        self._invocation_result = None  # The result of a DALI operator invocation.
        copied = False
        if tensors is not None:
            if isinstance(tensors, (_backend.TensorListCPU, _backend.TensorListGPU)):
                backend_dev = _backend_device(tensors)
                if (
                    (device is None or device == backend_dev)
                    and (dtype is None or dtype.type_id == tensors.dtype)
                    and (layout is None or layout == tensors.layout())
                ):
                    self._storage = tensors
                    self._device = backend_dev
                    self._layout = tensors.layout()
                    self._dtype = _dtype(tensors.dtype)
                else:
                    tmp = Batch(tensors)
                    if device is not None and device != tmp.device:
                        tmp = tmp.to_device(device)
                        copied = True
                    if dtype is not None and dtype != tmp.dtype:
                        from . import cast

                        tmp = cast(tmp, dtype=dtype, device=device)
                        copied = True
                    self._assign(tmp)
                    if self._storage and layout:
                        self._storage.set_layout(layout)
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
                if t._storage is not None:
                    if isinstance(t._storage, _backend.TensorCPU):
                        self._storage = _backend.TensorListCPU(t._storage, layout=layout)
                    elif isinstance(t._storage, _backend.TensorGPU):
                        self._storage = _backend.TensorListGPU(t._storage, layout=layout)
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
                        if not isinstance(t, Tensor) or t._storage is not sample._storage:
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
                    with device:
                        t = Tensor([], dtype=dtype, device=device).evaluate()
                        if self._device.device_type == "cpu":
                            backend_type = _backend.TensorListCPU
                        elif self._device.device_type == "gpu":
                            backend_type = _backend.TensorListGPU
                        else:
                            raise ValueError(
                                f"Internal error: "
                                f"Unsupported device type: {self._device.device_type}"
                            )
                        self._storage = backend_type(t._storage, layout=layout)

        if self._dtype is None:
            if self._storage is not None:
                self._dtype = DType.from_type_id(self._storage.dtype)
            else:
                self._dtype = dtype
        if self._device is None:
            if self._storage is not None:
                self._device = _backend_device(self._storage)
            else:
                self._device = device
        self._layout = layout
        if self._invocation_result is None:
            self._invocation_result = invocation_result
        else:
            assert invocation_result is None or invocation_result is self._invocation_result
        self._ndim = None
        if self._tensors and self._tensors[0]._shape:
            self._ndim = len(self._tensors[0]._shape)

        if copy and not copied:
            dev = self.to_device(self.device, force_copy=True)
            if dtype is not None and dev.dtype != dtype:
                from . import cast

                dev = cast(dev, dtype=dtype, device=device)
            self._assign(dev.evaluate())
            copied = True
        else:
            if self._dtype is not None and dtype is not None and self._dtype != dtype:
                from . import cast

                self._assign(cast(self, dtype=dtype, device=device))

        if _eval_mode.EvalMode.current().value >= _eval_mode.EvalMode.eager.value:
            self.evaluate()

    def _is_external(self) -> bool:
        return self._wraps_external_data

    @staticmethod
    def broadcast(
        sample,
        batch_size: int,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
    ) -> "Batch":
        """
        Creates a batch by repeating a single `sample` `batch_size` times.

        This function returns a batch obtained by repeating the sample `sample` `batch_size` times.
        Optionally, the result may be placed on the specified device (otherwise it will inherit the
        device from the `sample` argument) or converted to the desired data type.

        This function yields result equivalent to
        ``as_batch([tensor(sample, dtype=dtype, device=device)] * batch_size)``
        but is much more efficient.
        """
        if isinstance(sample, Batch):
            raise ValueError("Cannot broadcast a Batch")
        if _is_tensor_type(sample):
            t = _as_tensor(sample, device=device, dtype=dtype).evaluate()
            if t.device.device_type == "gpu":
                tl_type = _backend.TensorListGPU
            else:
                tl_type = _backend.TensorListCPU
            return Batch(tl_type.broadcast(t._storage, batch_size))
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
            if dtype is not None and dtype.kind != DType.Kind.enum:
                arr = arr.astype(_dali_types.to_numpy_type(dtype.type_id))
            arr = np.repeat(arr[np.newaxis], batch_size, axis=0)

        with nvtx.annotate("to backend", domain="batch"):
            tl = _backend.TensorListCPU(arr)
            if converted_dtype_id is not None:
                tl.reinterpret(converted_dtype_id)
        with nvtx.annotate("create batch", domain="batch"):
            return Batch(tl, device=device, dtype=dtype)

    @property
    def dtype(self) -> DType:
        """
        The element type of the tensors in the batch.
        """
        if self._dtype is None:
            if self._storage is not None:
                self._dtype = DType.from_type_id(self._storage.dtype)
            elif self._invocation_result is not None:
                self._dtype = _dtype(self._invocation_result.dtype)
            elif self._tensors:
                self._dtype = self._tensors[0].dtype
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        return self._dtype

    @property
    def device(self) -> Device:
        """
        The device on which the batch resides (or will reside, in case of lazy evaluation).
        """
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
        """
        The layout of tensors in the batch.

        The "batch dimension" (commonly denoted as N) is not included - a batch of HWC images
        will have HWC layout, not NHWC.
        """
        if self._layout is None:
            if self._invocation_result is not None:
                self._layout = self._invocation_result.layout
            elif self._storage is not None:
                self._layout = self._storage.layout()
                if self._layout == "" and self.ndim != 0:
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
        """
        The number of dimensions of the samples in the batch.

        The "batch dimension" is not included - e.g. a batch of HWC is still a 3D object.
        """
        if self._ndim is None:
            if self._storage is not None:
                self._ndim = self._storage.ndim()
            elif self._invocation_result is not None:
                self._ndim = self._invocation_result.ndim
            elif self._tensors:  # not None and not empty
                self._ndim = self._tensors[0].ndim
            else:
                raise ValueError("Cannot establish the number of dimensions of an empty Batch")
        return self._ndim

    @property
    def tensors(self):
        """
        Returns an indexable list of :class:`Tensor` objects that comprise the batch.
        """
        return _TensorList(self)

    def to_device(self, device: Device, force_copy: bool = False) -> "Batch":
        """
        Returns the data batch on the specified device.

        If the batch already resides on the device specified, the function will return `self`
        unless a copy is explicitly requested by passing ``force_copy=True``
        """
        if device is not None and not isinstance(device, Device):
            device = _device(device)
        if self.device == device and not force_copy:
            return self
        else:
            copy_dev = device if device.device_type == "gpu" else self.device
            with copy_dev:
                from . import copy

                return copy(self, device=device)

    def cpu(self) -> "Batch":
        """
        Returns the batch on the CPU. If it's already there, this function returns `self`.
        """
        return self.to_device(Device("cpu"))

    def gpu(self, index: Optional[int] = None) -> "Batch":
        """
        Returns the batch on the GPU. If it's already there, this function returns `self`.

        If index is not specified, the current CUDA device is used.
        """
        return self.to_device(Device("gpu", index))

    def _assign(self, other: "Batch"):
        if other is self:
            return
        self._device = other._device
        self._dtype = other._dtype
        self._layout = other._layout
        self._storage = other._storage
        if other._tensors is not None:
            self._tensors = [t for t in other._tensors]  # copy the list
        else:
            self._tensors = None
        self._invocation_result = other._invocation_result
        self._wraps_external_data = other._wraps_external_data

    @property
    def slice(self):
        """Interface for samplewise slicing.

        Regular slicing selects samples first and then slices each sample with common
        slicing parameters.

        Samplewise slicing interface allows the slicing parmaters to be batches (with the same
        number of samples) and the slicing parameters are applied to respective samples.

        ::

            start = Batch([1, 2, 3])
            stop = Batch([4, 5, 6])
            step = Batch([1, 1, 2])
            sliced = input.slice[start, stop, step]
            # the result is equivalent to
            sliced = Batch([
                sample[start[i]:stop[i]:step[i]]
                for i, sample in enumerate(input)
            ])

        If the slicing parameters are not batches, they are broadcast to all samples.
        """
        return BatchedSlice(self)

    def __iter__(self):
        """
        Iterates over tensors in the batch.
        """
        return iter(self.tensors)

    def select(self, sample_range):
        """
        Selects a range of samples.

        The result of this function is either a :class:`Batch` (if `sample_range` is a `slice` or a
        `list`) or a :class:`Tensor` if `sample_range` is a number.
        """
        r = sample_range
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
            if self._storage:
                t._storage = self._storage[i]
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
        """
        The number of tensors in the batch.
        """
        if self._storage is not None:
            return len(self._storage)
        elif self._tensors is not None:
            return len(self._tensors)
        elif self._invocation_result is not None:
            return self._invocation_result.batch_size
        else:
            raise ValueError("Neither tensors nor invocation result are set")

    @property
    def shape(self):
        """
        The shape of the batch.

        Returns the list of shapes of individual samples.

        Example::

            >>> import nvidia.dali.experimental.dynamic as ndd
            >>> import numpy as np
            >>> t0 = ndd.tensor(np.zeros((480, 640, 3)))
            >>> t1 = ndd.tensor(np.zeros((720, 1280, 1)))
            >>> b = ndd.as_batch([t0, t1])
            >>> print(b.shape)
            [(480, 640, 3), (720, 1280, 1)]
        """
        if self._invocation_result is not None:
            return self._invocation_result.shape
        if self._storage is not None:
            return self._storage.shape()
        else:
            assert self._tensors is not None
            return [t.shape for t in self._tensors]

    def __str__(self) -> str:
        return "Batch(\n" + str(self.evaluate()._storage) + ")"

    def evaluate(self):
        """
        Evaluates the underlying lazy expression, if any.

        If the batch is a result of a lazy evaluation, calling `evaluate` will cause the expression
        to be evaluated. If the batch already contains concrete data, this function has no effect.

        The behavior of this function is affected by the current evaluation context and current
        device. See :class:`EvalContext` and :class:`Device` for details.

        The function returns `self`.
        """
        if self._storage is None:
            # TODO(michalz): Consider thread-safety
            if self._invocation_result is not None:
                self._storage = self._invocation_result.value()
            else:
                with self._device:
                    if self._device.device_type == "cpu":
                        backend_type = _backend.TensorListCPU
                    elif self._device.device_type == "gpu":
                        backend_type = _backend.TensorListGPU
                    else:
                        raise ValueError(
                            f"Internal error: "
                            f"Unsupported device type: {self._device.device_type}"
                        )
                    self._storage = backend_type(
                        [t.evaluate()._storage for t in self._tensors], self.layout
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
    """Constructs a :class:`Batch` object.

    Constructs a batch by copying the input tensors and optionally converting them to the desired
    data type and storing on the specified device.

    Parameters
    ----------
    tensors : TensorLike, default: None
        The data to construct the batch from. Can be a list of tensors, a TensorList,
        or other supported types.
        Supported types are:

        - a :class:`Batch` object; the batch is copied and the data is converted and moved to the
          specified device, if necessary
        - a list of tensor-like objects; the objects need to have matching number of dimensions,
          data types and layouts,
        - a tensor-like object; the outermost dimenion is interpreted as the batch dimension
        - a dali.backend.TensorListCPU or dali.backend.TensorListGPU
    dtype : DType, default: None
        The desired data type of the batch. If not specified, the data type is inferred
        from the input tensors. If specified, the input tensors are cast to the desired data type.
        The `dtype` is required if tensors are an empty list.
    device : Device or str, optional, default: None
        The device on which the batch should reside (e.g., "cpu" or "gpu").
        If not specified, the device is inferred from the input tensors.
    layout : str, optional, default: None
        The layout string describing the dimensions of the batch (e.g., "HWC").
        If not specified, the layout is inferred from the input tensors.
    """
    if isinstance(tensors, Batch):
        b = tensors.to_device(device or tensors.device, force_copy=True)
        if dtype is not None and b.dtype != dtype:
            from . import cast

            b = cast(b, dtype=dtype, device=device)
        return b.evaluate()
    else:
        return Batch(tensors, dtype=dtype, device=device, layout=layout, copy=True)


def as_batch(
    tensors: Union[Batch, Sequence[Any]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    layout: Optional[str] = None,
):
    """Constructs a :class:`Batch` object, avoiding the copy.

    Constructs a batch by viewing the input tensors as a batch. If the input tensors do not
    reside on the specified device or do not match the desired type, the data will be converted
    and/or copied, as necessary.

    Parameters
    ----------
    tensors : TensorLike, default: None
        The data to construct the batch from. It can be a list of tensors, a TensorList,
        or other supported types. In general, the input tensors must be kept alive by the caller
        until the batch is no longer needed.
        Supported types are:

        - a :class:`Batch` object; the batch is copied and the data is converted and moved to the
          specified device, if necessary
        - a list of tensor-like objects; the objects need to have matching number of dimensions,
          data types and layouts,
        - a tensor-like object; the outermost dimenion is interpreted as the batch dimension
        - a dali.backend.TensorListCPU or dali.backend.TensorListGPU
    dtype : DType, default: None
        The desired data type of the batch. If not specified, the data type is inferred
        from the input tensors. If specified, the input tensors are cast to the desired data type.
        The `dtype` is required if `tensors` are an empty list.
    device : Device or str, optional, default: None
        The device on which the batch should reside (e.g., "cpu" or "gpu").
        If not specified, the device is inferred from the input tensors.
    layout : str, optional, default: None
        The layout string describing the dimensions of the batch (e.g., "HWC").
        If not specified, the layout is inferred from the input tensors.
    """
    if isinstance(tensors, Batch):
        b = tensors
        if device is not None:
            b = tensors.to_device(device)
        if dtype is not None and b.dtype != dtype:
            from . import cast

            b = cast(b, dtype=dtype, device=device)
        return b
    else:
        return Batch(tensors, dtype=dtype, device=device, layout=layout)


__all__ = ["Batch", "batch", "as_batch"]
