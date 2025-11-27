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

from typing import Any, Optional, Tuple, Union
from ._type import DType, dtype as _dtype, type_id as _type_id
from ._device import Device, device as _device
import nvidia.dali.backend as _backend
from ._eval_context import EvalContext as _EvalContext
from . import _eval_mode
from . import _invocation
import copy
import nvidia.dali.types


def _volume(shape: Tuple[int, ...]) -> int:
    ret = 1
    for s in shape:
        ret *= s
    return ret


def _backend_device(backend: Union[_backend.TensorCPU, _backend.TensorGPU]) -> Device:
    if isinstance(backend, _backend.TensorCPU):
        return Device("cpu")
    elif isinstance(backend, _backend.TensorGPU):
        return Device("gpu", backend.device_id())
    else:
        raise ValueError(f"Unsupported backend type: {type(backend)}")


def _get_array_interface(data):
    if a_func := getattr(data, "__array__", None):
        try:
            return a_func()
        except TypeError:  # CUDA torch tensor, CuPy array, etc.
            return None
    else:
        return None


def _try_convert_enums(arr):
    assert arr.dtype == object
    if arr.size == 0:
        raise ValueError("Cannot convert an empty array of `object` type.")
    item = arr.flat[0]
    import numpy as np

    if isinstance(item, nvidia.dali.types.DALIInterpType):
        return arr.astype(np.int32), nvidia.dali.types.INTERP_TYPE
    elif isinstance(item, nvidia.dali.types.DALIDataType):
        return arr.astype(np.int32), nvidia.dali.types.DATA_TYPE
    elif isinstance(item, nvidia.dali.types.DALIImageType):
        return arr.astype(np.int32), nvidia.dali.types.IMAGE_TYPE
    else:
        raise TypeError("Unexpected element type f{type(item)}")


class Tensor:
    """A Tensor object.

    This class represents a single tensor usable with DALI dynamic API. It can contain any of the
    following:

    - tensor data owned by DALI
    - external tensor data wrapped into a DALI tensor
    - a sample taken out of a Batch object
    - a result of a lazy evaluation of a DALI operator

    In case of lazy evaluation, the operations are executed only after an attempt is made to access
    the tensor data or properties which cannot be obtained without running the underlying operation.
    """

    def __init__(
        self,
        data: Optional[Any] = None,
        dtype: Optional[Any] = None,
        device: Optional[Device] = None,
        layout: Optional[str] = None,
        batch: Optional[Any] = None,
        index_in_batch: Optional[int] = None,
        invocation_result: Optional[_invocation.InvocationResult] = None,
        copy: bool = False,
    ):
        """Constructs a :class:`Tensor` object.

        .. warning::
            :class:`Tensor` objects should not be constructed directly, use :meth:`tensor` or
            :meth:`as_tensor` instead.

        The :class:`Tensor` object can be created either from an existing object, passed as `data`
        or from an invocation result.
        Unless explicitly requested with the `copy` parameter, this constructor will make best
        effort to avoid the copy.

        Parameters
        ----------
        data : TensorLike, default: None
            The data to construct the tensor from. It can be a tensor-like object, a (nested) list,
            TensorCPU/TensorGPU or other supported type.
        dtype : DType, default: None
            The desired data type of the tensor. If not specified, the data type is inferred
            from the input data. If specified, the input data is cast to the desired data type.
        device : Device or str, optional, default: None
            The device on which the tensor should reside (e.g., "cpu" or "gpu").
            If not specified, the device is inferred from the input data.
        layout : str, optional, default: None
            The layout string describing the dimensions of the tensor (e.g., "HWC").
            If not specified, the layout is inferred from the input data, if possible.
        batch : Batch, optional, default: None
            Use if the tensor is a view of a sample in a batch. Used together with `index_in_batch`.
        index_in_batch : int, optional, default: None
            The index of the tensor in the batch. Used together with `batch`.
        invocation_result : _invocation.InvocationResult, default: None
            The result of a DALI operator invocation, used for lazy evaluation
        copy : bool, optional, default: False
            If True, the input data is copied. If False, the constructor will avoid
            copying data when possible.
        """
        if layout is None:
            layout = ""
        elif not isinstance(layout, str):
            raise ValueError(f"Layout must be a string, got {type(layout)}")

        self._slice = None
        self._storage = None  # The backing storage of the tensor, TensorCPU or TensorGPU.
        self._batch = batch  # Used only if the tensor is a view of a sample in a batch.
        self._index_in_batch = index_in_batch  # likewise
        self._invocation_result = None  # The result of a DALI operator invocation.
        self._device = None
        self._shape = None
        self._dtype = None
        self._layout = None
        self._wraps_external_data = False

        if device is not None and not isinstance(device, Device):
            device = _device(device)

        copied = False

        if dtype is not None:
            if not isinstance(dtype, DType):
                dtype = _dtype(dtype)

        if batch is not None:
            from . import _batch

            if not isinstance(batch, _batch.Batch):
                raise ValueError("The `batch` argument must be a `Batch`")
            self._batch = batch
            self._index_in_batch = index_in_batch
            self._dtype = batch.dtype
            self._device = batch.device
            self._layout = batch.layout
        elif data is not None:
            if isinstance(data, (_backend.TensorCPU, _backend.TensorGPU)):
                self._storage = data
                self._wraps_external_data = True
                self._device = _backend_device(data)
            elif isinstance(data, Tensor):
                if dtype is None or _type_id(dtype) == data.dtype.type_id:
                    if device is None or device == data.device:
                        self._assign(data)
                        self._wraps_external_data = data._wraps_external_data
                    else:
                        dev = data.to_device(device).evaluate()
                        if dev is not self:
                            copied = True
                        self._assign(dev)
                        self._wraps_external_data = not copied
                else:
                    from . import cast

                    if device is None:
                        device = data.device
                    converted = cast(data.to_device(device), dtype=dtype, device=device)
                    self._assign(converted.evaluate())
                    copied = True
            elif isinstance(data, TensorSlice):
                self._slice = data
            elif hasattr(data, "__dlpack_device__"):
                dl_device_type, device_id = data.__dlpack_device__()
                if int(dl_device_type) == 1 or int(dl_device_type) == 3:  # CPU
                    self._storage = _backend.TensorCPU(data.__dlpack__(), layout)
                elif int(dl_device_type) == 2:  # GPU
                    # If the current context is on the same device, use the same stream.
                    ctx = _EvalContext.current()
                    if ctx.device_id == device_id:
                        stream = ctx.cuda_stream
                    else:
                        stream = _backend.Stream(device_id)
                    args = {"stream": stream.handle}
                    self._storage = _backend.TensorGPU(
                        data.__dlpack__(**args),
                        layout=layout,
                        stream=stream,
                    )
                else:
                    raise ValueError(f"Unsupported device type: {dl_device_type}")
                self._wraps_external_data = True
            elif a := _get_array_interface(data):
                self._storage = _backend.TensorCPU(a, layout)
                self._wraps_external_data = True
            else:
                import numpy as np

                if dtype is not None:
                    # TODO(michalz): Built-in enum handling
                    self._storage = _backend.TensorCPU(
                        np.array(data, dtype=nvidia.dali.types.to_numpy_type(dtype.type_id)),
                        layout,
                        False,
                    )
                    copied = True
                    self._wraps_external_data = False
                    self._dtype = dtype
                else:
                    arr = np.array(data)
                    # DALI doesn't support int64 and float64, so we need to convert them to int32
                    # and float32, respectively.
                    converted_dtype_id = None
                    if arr.dtype == np.int64:
                        arr = arr.astype(np.int32)
                    elif arr.dtype == np.uint64:
                        arr = arr.astype(np.uint32)
                    elif arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
                    elif arr.dtype == object:
                        (arr, converted_dtype_id) = _try_convert_enums(arr)
                    self._storage = _backend.TensorCPU(arr, layout, False)
                    if converted_dtype_id is not None:
                        self._storage.reinterpret(converted_dtype_id)
                    copied = True
                    self._wraps_external_data = False

            if self._storage is not None:
                self._device = _backend_device(self._storage)
                if device is None:
                    device = self._device
            else:
                if self._device is None:
                    if device is None:
                        device = Device("cpu")
                    self._device = device
                else:
                    if device is None:
                        device = self._device

            if self._storage is not None:
                self._shape = tuple(self._storage.shape())
                self._dtype = DType.from_type_id(self._storage.dtype)
                self._layout = self._storage.layout()

            if self._storage is not None and device != _backend_device(self._storage):
                self._assign(self.to_device(device).evaluate())
                copied = True
        elif invocation_result is not None:
            self._invocation_result = invocation_result
            self._device = invocation_result.device
        else:
            raise ValueError("Either data, expression or batch and index must be provided")

        if dtype is not None and self._dtype != dtype:
            from . import cast

            self._assign(cast(self, dtype=dtype, device=self.device).evaluate())
            copied = True

        if _eval_mode.EvalMode.current().value >= _eval_mode.EvalMode.eager.value:
            self.evaluate()

        if copy and self._storage is not None and not copied:
            self._assign(self.to_device(device, True).evaluate())

    def _is_external(self) -> bool:
        return self._wraps_external_data

    def cpu(self) -> "Tensor":
        """
        Returns the tensor on the CPU. If it's already there, this function returns `self`.
        """
        return self.to_device(Device("cpu"))

    def gpu(self, index: Optional[int] = None) -> "Tensor":
        """
        Returns the tensor on the GPU. If it's already there, this function returns `self`.

        If index is not specified, the current CUDA device is used.
        """
        return self.to_device(Device("gpu", index))

    @property
    def device(self) -> Device:
        """
        The device on which the tensor resides (or will reside, in case of lazy evaluation).
        """
        if self._device is not None:
            return self._device
        if self._invocation_result is not None:
            self._device = self._invocation_result.device
            return self._device
        else:
            raise RuntimeError("Device not set")

    def to_device(self, device: Device, force_copy: bool = False) -> "Tensor":
        """
        Returns the tensor on the specified device.

        If the tensor already resides on the device specified, the function will return `self`
        unless a copy is explicitly requested by passing ``force_copy=True``
        """
        if self.device == device and not force_copy:
            return self
        else:
            copy_dev = device if device.device_type == "gpu" else self.device
            with copy_dev:
                from . import copy

                return copy(self, device=device)

    def _assign(self, other: "Tensor"):
        if other is self:
            return
        self._device = other._device
        self._shape = other._shape
        self._dtype = other._dtype
        self._layout = other._layout
        self._storage = other._storage
        self._slice = other._slice
        self._batch = other._batch
        self._index_in_batch = other._index_in_batch
        self._invocation_result = other._invocation_result
        self._wraps_external_data = other._wraps_external_data

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the tensor.

        A 0D tensor is a scalar and cannot be empty (it always contains a single value).
        Tensors with higher `ndim` can be empty if any of the extents is 0.
        """
        if self._storage is not None:
            return self._storage.ndim()
        elif self._slice is not None:
            return self._slice.ndim
        elif self._invocation_result is not None:
            return self._invocation_result.ndim
        elif self._batch is not None:
            return self._batch.ndim
        else:
            raise RuntimeError("Cannot determine the number of dimensions of the tensor.")

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the tensor, returned as a tuple of integers.
        """
        if self._shape is None:
            if self._invocation_result is not None:
                self._shape = self._invocation_result.shape
            elif self._slice:
                self._shape = self._slice.shape
            elif self._batch is not None:
                self._shape = self._batch.shape[self._index_in_batch]
            else:
                self._shape = tuple(self._storage.shape())
        return self._shape

    @property
    def dtype(self) -> DType:
        """
        The type of the elements of the tensor.
        """
        if self._dtype is None:
            if self._invocation_result is not None:
                self._dtype = _dtype(self._invocation_result.dtype)
            elif self._slice:
                self._dtype = self._slice.dtype
            elif self._batch is not None:
                self._dtype = self._batch.dtype
            else:
                self._dtype = _dtype(self._storage.dtype)
        return self._dtype

    @property
    def layout(self) -> str:
        """
        The semantic layout of the tensor, e.g. HWC, CHW.

        The layout assigns meaning to the axes. It affects the way in which the data is
        interpreted by some operators.

        Image/video/volume layouts:
        H - height,
        W - width,
        D - depth,
        C - channels,
        F - frames

        Audio layouts:
        f - frequency
        t - time
        C - channels
        """
        if self._layout is None:
            if self._invocation_result is not None:
                self._layout = self._invocation_result.layout
            elif self._slice:
                self._layout = self._slice.layout
            elif self._batch is not None:
                self._layout = self._batch.layout
            else:
                self._layout = self._storage.layout()
        # Use "" to indicate that the layout has been checked and is empty, but still return None
        # to avoid situations where we return a string with a length that doesn't match the number
        # of dimensions.
        return self._layout or None

    @property
    def size(self) -> int:
        """
        The number of elements in the tensor.
        """
        return _volume(self.shape)

    @property
    def nbytes(self) -> int:
        """
        The number of bytes required to store all elements in the tensor assuming dense packing.
        """
        return self.size * self.dtype.bytes

    @property
    def itemsize(self) -> int:
        """
        The size, in bytes, of a single element.
        """
        return self.dtype.bytes

    def item(self) -> Any:
        """
        Returns the only item in the tensor. Useful for scalars (0D tensors).
        """
        if self.size != 1:
            raise ValueError(f"Tensor has {self.size} elements, expected 1")
        import numpy as np

        with _EvalContext.current():
            return np.array(self.cpu().evaluate()._storage).item()

    def __array__(self):
        b = self.evaluate()._storage
        if isinstance(b, _backend.TensorCPU):
            import numpy as np

            return np.array(b)
        else:
            raise TypeError("This is not a CPU tensor. Use `.cpu()` to get the array interface.")

    @property
    def __cuda_array_interface__(self):
        b = self.evaluate()._storage
        if isinstance(b, _backend.TensorGPU):
            return b.__cuda_array_interface__
        else:
            raise TypeError(
                "This is not a GPU tensor. Use `.gpu()` to get the CUDA array interface."
            )

    def evaluate(self):
        """
        Evaluates the underlying lazy expression, if any.

        If the tensor is a result of a lazy evaluation, calling `evaluate` will cause the expression
        to be evaluated. If the tensor already contains concrete data, this function has no effect.

        The behavior of this function is affected by the current evaluation context and current
        device. See :class:`EvalContext` and :class:`Device` for details.

        The function returns `self`.
        """
        if self._storage is None:
            # TODO(michalz): Consider thread-safety
            if self._slice:
                self._storage = self._slice.evaluate()._storage
            elif self._batch is not None:
                t = self._batch._tensors[self._index_in_batch]
                if t is self:
                    self._storage = self._batch.evaluate()._storage[self._index_in_batch]
                else:
                    self._storage = t.evaluate()._storage
            else:
                assert self._invocation_result is not None
                self._storage = self._invocation_result.value()
            self._shape = tuple(self._storage.shape())
            self._dtype = DType.from_type_id(self._storage.dtype)
            self._layout = self._storage.layout()
        return self

    def __getitem__(self, ranges: Any) -> "Tensor":
        if not isinstance(ranges, tuple):
            ranges = (ranges,)

        if all(_is_full_slice(r) or r is Ellipsis for r in ranges):
            return self
        else:
            if self._slice:
                return self._slice.__getitem__(ranges)
            else:
                return Tensor(TensorSlice(self, ranges))

    def _is_same_tensor(self, other: "Tensor") -> bool:
        return (
            self._storage is other._storage
            and self._invocation_result is other._invocation_result
            and self._slice is other._slice
        )

    def __str__(self) -> str:
        return "Tensor(\n" + str(self.evaluate()._storage) + ")"

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


def _arithm_op(name, *args, **kwargs):
    argsstr = " ".join(f"&{i}" for i in range(len(args)))
    from . import arithmetic_generic_op

    return arithmetic_generic_op(*args, expression_desc=f"{name}({argsstr})")


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
    return not isinstance(r, slice) and r is not Ellipsis


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
    def __init__(self, tensor: Tensor, ranges: Tuple[Any, ...], absolute=False):
        self._tensor = copy.copy(tensor)
        self._ndim_dropped = 0
        self._shape = None
        if absolute:
            self._absolute_ranges = [copy.copy(r) for r in ranges]
            self._ranges = self._insane_pythonic_ranges(self._absolute_ranges, tensor.shape)
        else:
            self._ranges = [copy.copy(r) for r in ranges]
            self._absolute_ranges = None
        self._layout = None
        num_ranges = len(ranges)
        ellipsis_found = False
        for r in ranges:
            if _is_index(r):
                self._ndim_dropped += 1
            elif r is Ellipsis:
                if ellipsis_found:
                    raise ValueError("Only one Ellipsis is allowed.")
                num_ranges -= 1
                ellipsis_found = True
        if num_ranges > tensor.ndim:
            raise ValueError(
                f"Number of ranges ({num_ranges}) "
                f"is greater than the number of dimensions of the tensor ({tensor.ndim})"
            )

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
                    if r.step < 0:
                        shape.append((r.stop + r.step - r.start + 1) // r.step)
                    else:
                        shape.append((r.stop + r.step - r.start - 1) // r.step)
            self._shape = tuple(shape)
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._tensor.dtype

    @property
    def device(self) -> Device:
        return self._tensor.device

    @property
    def layout(self) -> str:
        if self._layout is not None:
            return self._layout
        input_layout = self._tensor.layout
        if self._ndim_dropped == 0 or input_layout == "" or input_layout is None:
            self._layout = input_layout
            return self._layout

        j = 0
        layout = ""
        for i, r in enumerate(self._ranges):
            if isinstance(r, slice):
                layout += input_layout[j]
                j += 1
            elif r is Ellipsis:
                j += self._tensor.ndim - len(self._ranges) + 1
            else:
                j += 1  # skip this dimension
        self._layout = layout
        return self._layout

    @staticmethod
    def _canonicalize_ranges(ranges, in_shape) -> Tuple[int, ...]:
        """Converts the ranges to sane non-pythonic values without negative indices wrapping"""
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
                if r.start is not None:
                    start = _scalar_value(r.start)
                    if start < 0:
                        start += extent
                else:
                    start = extent - 1 if step < 0 else 0
                if r.stop is not None:
                    stop = _scalar_value(r.stop)
                    if stop < 0:
                        stop += extent
                else:
                    stop = -1 if step < 0 else extent
                if step < 0:
                    stop = _clamp(stop, -1, extent - 1)
                    start = _clamp(start, stop, extent)
                else:
                    start = _clamp(start, 0, extent)
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

    @staticmethod
    def _insane_pythonic_ranges(abs_ranges, shape) -> Tuple[int, ...]:
        """Converts an absolute range into ranges as expected by Pythonic slicing API"""
        py_ranges = []
        for r, s in zip(abs_ranges, shape):
            if isinstance(r, slice):
                stop = r.stop
                # The exclusive `stop` for negative ranges could be -1, but it means
                # something else in Python - so we need skip over the whole length of the
                # array to make it really negative.
                if r.step < 0:
                    if stop < 0:
                        stop -= s
                py_ranges.append(slice(r.start, stop, r.step))
            else:
                py_ranges.append(r)
        return tuple(py_ranges)

    def __getitem__(self, ranges: Any) -> "Tensor":
        if not isinstance(ranges, tuple):
            ranges = (ranges,)

        if all(_is_full_slice(r) or r is Ellipsis for r in ranges):
            return Tensor(self)
        else:
            ranges = self._canonicalize_ranges(ranges, self.shape)
            abs_ranges = list(self._absolute_ranges)
            i = 0
            for d, r in enumerate(self._absolute_ranges):
                if isinstance(r, slice):
                    if isinstance(ranges[i], slice):
                        start = r.start + ranges[i].start * r.step
                        stop = r.start + ranges[i].stop * r.step
                        step = r.step * ranges[i].step
                        abs_ranges[d] = slice(start, stop, step)
                    else:
                        abs_ranges[d] = r.start + ranges[i] * r.step
                    i += 1
            result = TensorSlice(self._tensor, tuple(abs_ranges), True)
            if _eval_mode.EvalMode.current().value >= _eval_mode.EvalMode.eager.value:
                result.evaluate()
            return Tensor(result)

    def evaluate(self):
        with _EvalContext.current():
            if len(self._ranges) == 0:
                return self._tensor.evaluate()

            if all(_is_full_slice(r) for r in self._ranges):
                return self._tensor.evaluate()

            args = {}
            d = 0
            for i, r in enumerate(self._ranges):
                if r is Ellipsis:
                    d = self._tensor.ndim - len(self._ranges) + i + 1
                elif isinstance(r, slice):
                    if r.start is not None:
                        args[f"lo_{d}"] = r.start
                    if r.stop is not None and r.stop >= 0:
                        args[f"hi_{d}"] = r.stop
                    if r.step is not None:
                        args[f"step_{d}"] = r.step
                    d += 1
                else:
                    args[f"at_{d}"] = r
                    d += 1

            from . import tensor_subscript

            return tensor_subscript(self._tensor, **args).evaluate()


def tensor(
    data: Any,
    dtype: Optional[Any] = None,
    device: Optional[Device] = None,
    layout: Optional[str] = None,
):
    """Copies an existing tensor-like object into a DALI tensor.

    Parameters
    ----------
    data : TensorLike, default: None
        The data to construct the tensor from. It can be a tensor-like object, a (nested) list,
        TensorCPU/TensorGPU or other supported type.
        Supported types are:

        - numpy arrays
        - torch tensors
        - types exposing __dlpack__ or __array__ interface
        - existing :class:`Tensor` objects
    dtype : DType, default: None
        The desired data type of the tensor. If not specified, the data type is inferred
        from the input data. If specified, the input data is cast to the desired data type.
    device : Device or str, optional, default: None
        The device on which the tensor should reside (e.g., "cpu" or "gpu").
        If not specified, the device is inferred from the input data.
    layout : str, optional, default: None
        The layout string describing the dimensions of the tensor (e.g., "HWC").
        If not specified, the layout is inferred from the input data, if possible.
    """
    return Tensor(data, dtype=dtype, device=device, layout=layout, copy=True)


def as_tensor(
    data: Any,
    dtype: Optional[Any] = None,
    device: Optional[Device] = None,
    layout: Optional[str] = None,
):
    """Wraps an existing tensor-like object into a DALI tensor.

    Parameters
    ----------
    data : TensorLike, default: None
        The data to construct the tensor from. It can be a tensor-like object, a (nested) list,
        TensorCPU/TensorGPU or other supported type.
        Supported types are:

        - numpy arrays
        - torch tensors
        - types exposing __dlpack__ or __array__ interface
        - existing :class:`Tensor` objects
    dtype : DType, default: None
        The desired data type of the tensor. If not specified, the data type is inferred
        from the input data. If specified, the input data is cast to the desired data type.
    device : Device or str, optional, default: None
        The device on which the tensor should reside (e.g., "cpu" or "gpu").
        If not specified, the device is inferred from the input data.
    layout : str, optional, default: None
        The layout string describing the dimensions of the tensor (e.g., "HWC").
        If not specified, the layout is inferred from the input data, if possible.
    """
    from . import _batch

    if isinstance(data, _batch.Batch):
        data = data.evaluate()._storage.as_tensor()

    return Tensor(data, dtype=dtype, device=device, layout=layout, copy=False)


__all__ = ["Tensor", "tensor", "as_tensor"]
