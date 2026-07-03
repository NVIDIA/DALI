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

import enum
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, cast

from nvidia.dali import fn

from ..._typing import BatchLike
from ..._utils.external_source_impl import get_callback_from_source
from . import _compile
from ._batch import Batch, _get_batch_size, as_batch
from ._device import DeviceLike
from ._device import device as _as_device
from ._nvtx import NVTXRange
from ._tensor import Tensor, as_tensor
from ._type import DTypeLike

if TYPE_CHECKING:
    from ._eval_context import EvalContext

_SourceOutput: TypeAlias = BatchLike | Sequence[BatchLike]
SourceType: TypeAlias = Callable[[], _SourceOutput] | Iterable[_SourceOutput]

_CallResult: TypeAlias = Tensor | Batch | tuple[Tensor, ...] | tuple[Batch, ...]


class _Role(enum.Enum):
    UNUSED = enum.auto()
    EAGER = enum.auto()
    FEEDER = enum.auto()  # pulled inside another source's compiled loop
    ROOT = enum.auto()  # iterated through its own .compiled()

    @property
    def is_compiled(self) -> bool:
        return self in (_Role.FEEDER, _Role.ROOT)


# We don't inherit from _ops.Operator because there's nothing to reuse from there
class ExternalSource:
    """Consume data from a Python callable or iterable source.

    The `source` can be either a callable or an iterable, returning a tensor-like or batch-like.
    An instance of this class is stateful; calling it pulls the next element(s) from the source.

    Parameters
    ----------

    source: callable or iterable
        The source of the data.

        The source is polled via ``source()`` or ``next(source)``. Data provided by `source`
        can be tensor-like, batch-like or a tuple thereof if `num_outputs` > 1.

    num_outputs : int, default: 1
        If specified, denotes the number of outputs produced by `source`.

    cycle : string or bool, optional
        Specifies if and how to cycle through the source. It can be one of the following values:

        -  ``"no"``, ``False`` or ``None`` - don't cycle; ``StopIteration`` is raised when
          end of data is reached; this is the default behavior
        - ``"quiet"`` or ``True`` - the data is repeated indefinitely,
        - ``"raise"`` - when the end of data is reached, ``StopIteration`` is raised, but
          the iteration is restarted on subsequent call.

        This flag requires that `source` is an iterable.

    device : device-like, default: "cpu"
        Device of the output data. If the device mismatches, this can cause implicit D2H/H2D copies.

    layout : :ref:`layout str<layout_str_doc>` or sequence thereof, optional
        Layout of the output data. May be a sequence of size `num_outputs`.

    dtype : dtype-like or sequence thereof, optional
        Data type of the output data. May be a sequence of size `num_outputs`.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>> import numpy as np

    An iterable source is consumed one element at a time:

    >>> es = ndd.ExternalSource([np.full((2, 2), i) for i in range(4)])
    >>> _ = es()  # skip the first one
    >>> es()
    Tensor(
        [[1 1]
         [1 1]],
        dtype=i64,
        device="cpu",
        shape=(2, 2))

    A sample output can be broadcast to a batch:

    >>> es = ndd.ExternalSource(lambda: np.arange(3))
    >>> es(batch_size=2)
    Batch(
        [[0 1 2],
        [0 1 2]],
        dtype=i64,
        device="cpu",
        num_samples=2,
        shape=[(3,), (3,)])

    With `num_outputs` > 1, a tuple is returned

    >>> es = ndd.ExternalSource(lambda: (np.zeros(4), np.ones(4)), num_outputs=2)
    >>> a, b = es()
    >>> b
    Tensor(
        [1. 1. 1. 1.],
        dtype=f64,
        device="cpu",
        shape=(4,))
    """

    def __init__(
        self,
        source: SourceType,
        num_outputs: int = 1,
        *,
        cycle: Literal["no", "quiet", "raise"] | bool | None = None,
        device: DeviceLike = "cpu",
        layout: str | Sequence[str] | None = None,
        dtype: DTypeLike | Sequence[DTypeLike] | None = None,
    ):
        callback, source_desc = get_callback_from_source(source, cycle)
        assert source_desc is not None  # `source` is never None here, so a callback is built
        if source_desc.has_inputs:
            raise ValueError("ndd.ExternalSource only supports callables with no parameters")
        self._callback = cast(Callable[[], _SourceOutput], callback)

        if num_outputs <= 0:
            raise ValueError("num_outputs must be strictly positive")
        self._num_outputs = num_outputs
        self._device = device
        self._layouts = self._broadcast_arg(layout)
        self._dtypes = self._broadcast_arg(dtype)

        self._role = _Role.UNUSED
        self._compile_source: _compile.CompileSource | None = None
        self._compiled_iter: _compile.CompiledEpochIterator | None = None

    @NVTXRange("__call__: ExternalSource", category="op_builder")
    def __call__(self, *, batch_size: int | None = None) -> _CallResult:
        """Consume one item from the source.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to broadcast output tensors to. Validated against batch outputs.

        Returns
        -------
        `Tensor`, `Batch`, or tuple thereof
            A `Batch` if the source produced a `Batch` or a TensorList, a `Tensor` otherwise.
            If `num_outputs` > 1, a tuple is returned.

        Raises
        ------
        StopIteration
            When the source is exhausted, depending on the ``cycle`` argument.
        """
        # Valid dispatch paths:
        # - without a compile context, run eagerly
        # - while tracing, pull and register the source as a feeder
        # - while executing a compiled context, return the traced feeder's result

        ctx = _compile.CompileContext.current()
        if ctx is None:
            if self._role.is_compiled:
                raise RuntimeError("This ExternalSource is already used in a compiled loop")
            self._role = _Role.EAGER
            return self._eager_call(batch_size=batch_size)

        if self._role is _Role.EAGER:
            raise RuntimeError("This ExternalSource was already used eagerly")
        if self._role is _Role.ROOT:
            raise RuntimeError("Instance already used through .compiled() method")

        if ctx.state is _compile.State.TRACING:
            result = self._trace_pull(ctx, batch_size)
            self._role = _Role.FEEDER
            return result
        return self._compiled_call(ctx, batch_size)

    def compiled(self, batch_size: int, ctx: "EvalContext | None" = None):
        """Iterate one epoch with this source as the compiled graph's root.

        ``ExternalSource`` equivalent of :meth:`Reader.next_epoch` with ``compile=True``.

        Any other ``ExternalSource`` called inside the loop must be consumed exactly once per step.
        They are prefetched, so they are polled ahead of the loop body and breaking out may discard
        already-pulled items.
        """
        if self._role is _Role.EAGER:
            raise RuntimeError("This ExternalSource was already used eagerly")
        if self._role is _Role.FEEDER:
            raise RuntimeError("Instance already used through __call__")

        iterator = _compile.make_iterator(self, batch_size)
        self._role = _Role.ROOT
        return iterator.batches(ctx)

    def _unwrap(self, outputs: tuple[Tensor, ...] | tuple[Batch, ...]) -> _CallResult:
        return outputs[0] if self._num_outputs == 1 else outputs

    def _eager_call(self, *, batch_size: int | None = None) -> _CallResult:
        data = self._callback()
        outputs = self._convert_outputs(data, batch_size)
        return self._unwrap(outputs)

    def _trace_pull(self, ctx: _compile.CompileContext, batch_size: int | None) -> _CallResult:
        """Pull, convert and wrap one item during tracing, registering the root on first use."""
        src = self._compile_source
        if src is not None and src.ctx is not ctx:
            raise RuntimeError("Already bound to a different compile context.")

        ctx.check_batch_size(batch_size)
        # pull before registering: an empty source raises here, leaving nothing half-bound
        try:
            data = self._callback()
            tensor_lists = self._to_tensor_lists(data, ctx.batch_size)
        except Exception:
            self._teardown_compile()
            raise

        if src is None:
            src = self._compile_source = ctx.add_source(self._num_outputs, self)
        ctx._mark_read(src)
        return self._unwrap(ctx._wrap_tensor_lists(src, tensor_lists))

    def _compiled_call(self, ctx: _compile.CompileContext, batch_size: int | None) -> _CallResult:
        src = self._compile_source
        if src is None or src.ctx is not ctx:
            ctx._teardown()
            raise RuntimeError("ExternalSource wasn't seen during tracing")

        ctx.check_batch_size(batch_size)
        ctx._mark_read(src)
        return ctx.result_for(src)

    def _source_callback(self):
        """``fn.external_source``'s ``source`` callback: pull, convert, return TensorList(s)"""
        src = self._compile_source
        assert src is not None
        try:
            data = self._callback()
        except StopIteration:
            src.ctx._mark_stopped(src)  # lets the loop tell a clean epoch end from underrun
            raise
        tensor_lists = self._to_tensor_lists(data, src.ctx.batch_size)
        return tensor_lists[0] if self._num_outputs == 1 else list(tensor_lists)

    def _to_tensor_lists(self, data: _SourceOutput, batch_size: int) -> tuple:
        outputs = self._convert_outputs(data, batch_size)
        return tuple(output.evaluate()._storage for output in outputs)

    def _teardown_compile(self):
        self._role = _Role.UNUSED
        self._compile_source = None
        self._compiled_iter = None

    def _wire_pipeline(self, source: "_compile.CompileSource") -> tuple:
        device = _as_device(self._device).device_type
        if source.num_outputs == 1:
            return (fn.external_source(self._source_callback, device=device),)
        out = fn.external_source(self._source_callback, source.num_outputs, device=device)
        return tuple(out)

    def _shape_result(self, source, batches: tuple):
        return self._unwrap(batches)

    def _transfer_into(self, pipe) -> bool:
        return False  # an ExternalSource holds no native op to move into a pipeline

    def _make_epoch_iterator(self, batch_size: int) -> "_compile.CompiledEpochIterator":
        return _compile._ExternalSourceEpochIterator(self, batch_size)

    def _convert_outputs(
        self, data: _SourceOutput, batch_size: int | None
    ) -> tuple[Tensor, ...] | tuple[Batch, ...]:
        """Convert the source's outputs, requiring them uniformly Tensors or uniformly Batches."""
        outputs = self._get_outputs(data)
        results = tuple(
            self._convert_output(output, batch_size, idx) for idx, output in enumerate(outputs)
        )
        if not _are_types_uniform(results):
            raise TypeError("Outputs must be uniformly Tensors or uniformly Batches")
        return results

    def _get_outputs(self, data: _SourceOutput) -> Sequence[BatchLike]:
        if self._num_outputs == 1:
            return (data,)  # type: ignore
        if not isinstance(data, Sequence) or len(data) != self._num_outputs:
            raise ValueError(f"Expected {self._num_outputs} outputs from the source")
        return data  # type: ignore

    def _convert_output(self, data: BatchLike, batch_size: int | None, idx: int) -> Tensor | Batch:
        layout = self._layouts[idx]
        dtype = self._dtypes[idx]
        device = _as_device(self._device)

        actual_batch_size = _get_batch_size(data)
        if actual_batch_size is not None:
            batch = as_batch(data, dtype=dtype, device=device, layout=layout)
            if batch_size is not None and actual_batch_size != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got {actual_batch_size}")
            return batch

        tensor = as_tensor(data, dtype=dtype, device=device, layout=layout)
        if batch_size is not None:
            return Batch.broadcast(tensor, batch_size=batch_size)
        return tensor

    def _broadcast_arg(self, value: Any | Sequence) -> Sequence:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return (value,) * self._num_outputs

        if len(value) != self._num_outputs:
            raise ValueError(f"Expected a sequence of size {self._num_outputs}, got {len(value)}")
        return value


def _are_types_uniform(
    values: tuple[Tensor | Batch, ...],
) -> TypeGuard[tuple[Tensor, ...] | tuple[Batch, ...]]:
    # We know that values[0] exists since _num_outputs > 0
    expected_type = Batch if isinstance(values[0], Batch) else Tensor
    return all(isinstance(value, expected_type) for value in values)
