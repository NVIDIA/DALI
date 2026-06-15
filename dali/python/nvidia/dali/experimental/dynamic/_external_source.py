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

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeAlias, cast, TypeGuard

from ..._typing import BatchLike
from ..._utils.external_source_impl import get_callback_from_source
from ._batch import Batch, _get_batch_size, as_batch
from ._device import DeviceLike
from ._nvtx import NVTXRange
from ._tensor import Tensor, as_tensor
from ._type import DTypeLike

# Note: TensorLike <: BatchLike
_SourceOutput: TypeAlias = BatchLike | Sequence[BatchLike]
SourceType: TypeAlias = Callable[[], _SourceOutput] | Iterable[_SourceOutput]


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

    @NVTXRange("__call__: ExternalSource", category="op_builder")
    def __call__(
        self, *, batch_size: int | None = None
    ) -> Tensor | Batch | tuple[Tensor, ...] | tuple[Batch, ...]:
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
        outputs = self._get_outputs(self._callback())
        results = tuple(
            self._convert_output(output, batch_size, idx) for idx, output in enumerate(outputs)
        )
        if not _are_types_uniform(results):
            raise TypeError("Outputs must be uniformly Tensors or uniformly Batches")
        return results[0] if self._num_outputs == 1 else results

    def _get_outputs(self, data: _SourceOutput) -> Sequence[BatchLike]:
        if self._num_outputs == 1:
            return (cast(BatchLike, data),)
        if not isinstance(data, Sequence) or len(data) != self._num_outputs:
            raise ValueError(f"Expected {self._num_outputs} outputs from the source")
        return data  # type: ignore

    def _convert_output(self, data: BatchLike, batch_size: int | None, idx: int) -> Tensor | Batch:
        layout = self._layouts[idx]
        dtype = self._dtypes[idx]

        actual_batch_size = _get_batch_size(data)
        if actual_batch_size is not None:
            batch = as_batch(data, dtype=dtype, device=self._device, layout=layout)
            if batch_size is not None and actual_batch_size != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got {actual_batch_size}")
            return batch

        tensor = as_tensor(data, dtype=dtype, device=self._device, layout=layout)
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
