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

import threading
from typing import TYPE_CHECKING, Any, Optional

import nvtx

from ._async import _Future
from ._device import Device
from ._eval_context import EvalContext as _EvalContext
from ._eval_mode import EvalMode as _EvalMode
from ._type import DType

if TYPE_CHECKING:
    from .ops import Operator


class Invocation:
    """
    A class representing a single invocation of an operator.

    It binds the operator instance and the call arguments.
    It also tracks the order of invocations of stateful operators, which is important for
    lazy evaluation of stateful operators or operators with side-effects.

    NOTE:  This class is not thread safe. Subsequent invocations of the same operator instance
           must be synchronized by the caller.
    """

    def __init__(
        self,
        operator_instance: "Operator",
        call_id: Optional[int],
        inputs: list[Any] | None = None,
        args: dict[str, Any] | None = None,
        is_batch: bool = False,
        batch_size: Optional[int] = None,
        previous_invocation: Optional["Invocation"] = None,
    ):
        """
        Parameters
        ----------
        operator_instance : OperatorInstance
            The operator instance that is being invoked.
        call_id : int
            The call ID of the invocation - necessary to avoid folding of multiple invocations
            of the same stateful operator.
        inputs : list
            The inputs to the operator.
        args : dict name->argument
            The argument inputs of the operator. Scalar arguments are part of the operator instance.
        is_batch : bool
            Whether this is a batch invocation.
            NOTE: A batch of 1 and a single tensor are equivalent from the operator's perspective
                  (operators always work with batches) but differes from the user's perspective.
        batch_size : int
            The batch sizes. This is useful chiefly for operators without inputs or ones that alter
            the batch size.
        previous_invocation : Invocation
            The previous invocation of the same operator. Used by stateful operators.
        """
        self._operator = operator_instance
        self._call_id = call_id
        self._inputs = inputs or []
        self._args = args or {}
        self._is_batch = is_batch
        self._results: tuple[Any] | None = None
        self._batch_size = batch_size
        self._num_outputs: int | None = None
        self._output_devices: list[Device] | None = None
        self._previous_invocation = previous_invocation
        self._eval_context = _EvalContext.current()
        self._future: Optional[_Future] = None
        self._run_lock = threading.Lock()

    def device(self, result_index: int):
        if self._output_devices is None:
            self._output_devices = self._operator._infer_output_devices(*self._inputs, **self._args)
        return self._output_devices[result_index]

    def ndim(self, result_index: int) -> int:
        if self._results is None:
            # TODO(michalz): Try to get ndim without full evaluation.
            self.run(self._eval_context)
        return self._results[result_index].ndim()

    def shape(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get shape without full evaluation.
            self.run(self._eval_context)
        s = self._results[result_index].shape()
        return s if self.is_batch else tuple(s)

    def dtype(self, result_index: int) -> DType:
        if self._results is None:
            # TODO(michalz): Try to get dtype without full evaluation.
            self.run(self._eval_context)
        return self._results[result_index].dtype

    def batch_size(self, result_index: int):
        with nvtx.annotate("Invocation.batch_size", domain="invocation"):
            if not self._is_batch:
                return None
            if self._batch_size is not None:
                return self._batch_size
            if self._results is None:
                # TODO(michalz): Try to get batch_size without full evaluation.
                self.run(self._eval_context)
            return self._results[result_index].batch_size if self._is_batch else None

    def layout(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get layout without full evaluation.
            self.run(self._eval_context)
        return self._results[result_index].layout()

    def __iter__(self):
        for index in range(len(self)):
            yield InvocationResult(self, index)

    def __getitem__(self, index):
        """
        Returns a proxy to the index-th result of the invocation.
        """
        return InvocationResult(self, index)

    def __len__(self) -> int:
        if self._num_outputs is None:
            self._num_outputs = self._operator._infer_num_outputs(*self._inputs, **self._args)
        return self._num_outputs  # type: ignore

    @property
    def is_batch(self):
        return self._is_batch

    def run(self, ctx: Optional[_EvalContext] = None):
        if future := self._future:
            with nvtx.annotate("Invocation.wait", domain="invocation"):
                future.wait()
            self._future = None
        else:
            ctx = self._eval_context if ctx is None else ctx
            # We don't want to run from two threads
            with self._run_lock:
                self._run_impl(ctx)

    def schedule(self, ctx: Optional[_EvalContext] = None):
        """Schedule the asynchronous execution of the operator"""

        # Note: this function can only be called once, soon after the instance creation
        # so we don't have to worry about thread-safety

        if self._results is not None or self._future is not None:
            return

        eval_mode = _EvalMode.current()
        device = Device.current()
        eval_context = self._eval_context if ctx is None else ctx

        def _run():
            # Forward thread-local context to the new thread
            with eval_context, eval_mode, device:
                self._run_impl(eval_context)

        self._future = eval_context._async_executor.submit(_run)

    def _run_impl(self, ctx: _EvalContext):
        """Run the operator and store the result in self._results"""

        if self._results is not None:
            return

        with nvtx.annotate("Invocation.run", domain="invocation"):
            # If the invocation was created with a GPU device, validate that
            # the evaluation context matches.
            if (
                self._eval_context._device is not None
                and self._eval_context._device.device_type == "gpu"
            ):
                if self._eval_context._device != ctx._device:
                    raise RuntimeError(
                        f"Device mismatch: Invocation was created with device "
                        f"'{self._eval_context._device}' "
                        f"but is being evaluated with device '{ctx._device}'. "
                        f"This can happen when using an EvalContext context manager "
                        f"at invocation time and a different EvalContext at evaluation "
                        f"time. Ensure consistent EvalContext usage."
                    )

            if self._previous_invocation is not None:
                self._previous_invocation.run(ctx)
                self._previous_invocation = None
            # TODO(michalz): Implement something that doesn't leak memory
            # cached = ctx.cached_results(self)
            # if cached is not None:
            #     self._results = cached
            # else:
            r = self._operator._run(
                ctx,
                *self._inputs,
                batch_size=self._batch_size if self._is_batch else None,
                **self._args,
            )
            if isinstance(r, (tuple, list)):
                self._results = tuple(r)
            elif isinstance(r, dict):
                self._results = tuple(r.values())
            else:
                self._results = (r,)
            ctx.cache_results(self, self._results)

    def values(self, ctx: Optional[_EvalContext] = None):
        """
        Returns the concrete results of the invocation.

        The invocation may have multiple results (e.g. readers may produce data + labels).
        The return value is a list of Batch or Tensor objects.
        """
        self.run(ctx)
        return self._results


class InvocationResult:
    """
    A class representing a single result of an invocation.

    It binds the invocation and the index of the return value.
    It serves as a proxy to enable lazy evaluation.
    """

    def __init__(self, invocation, index: int):
        self._invocation = invocation
        self._index = index

    @property
    def device(self):
        return self._invocation.device(self._index)

    @property
    def ndim(self) -> int:
        return self._invocation.ndim(self._index)

    @property
    def shape(self):
        return self._invocation.shape(self._index)

    @property
    def dtype(self) -> DType:
        return self._invocation.dtype(self._index)

    @property
    def layout(self):
        return self._invocation.layout(self._index)

    @property
    def batch_size(self):
        return self._invocation.batch_size(self._index)

    @property
    def is_batch(self):
        return self._invocation.is_batch

    def value(self, ctx: Optional[_EvalContext] = None):
        return self._invocation.values(ctx)[self._index]

    @property
    def invocation(self):
        return self._invocation

    @property
    def index(self):
        return self._index
