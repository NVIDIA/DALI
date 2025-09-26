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

from typing import Optional
from ._eval_context import EvalContext as _EvalContext
from ._type import DType
import nvtx


class Invocation:
    def __init__(
        self,
        operator_instance,
        call_id,
        inputs=[],
        args={},
        is_batch: bool = False,
        batch_size: Optional[int] = None,
        previous_invocation: Optional["Invocation"] = None,
    ):
        self._operator = operator_instance
        self._call_id = call_id
        self._inputs = inputs
        self._args = args
        self._is_batch = is_batch
        self._results = None
        self._batch_size = batch_size
        self._num_outputs = None
        self._output_devices = None
        self._previous_invocation = previous_invocation

    def device(self, result_index: int):
        if self._output_devices is None:
            self._output_devices = self._operator.infer_output_devices(*self._inputs, **self._args)
        return self._output_devices[result_index]

    def ndim(self, result_index: int) -> int:
        if self._results is None:
            # TODO(michalz): Try to get ndim without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].ndim()

    def shape(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get shape without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].shape()

    def dtype(self, result_index: int) -> DType:
        if self._results is None:
            # TODO(michalz): Try to get dtype without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].dtype

    def batch_size(self, result_index: int):
        with nvtx.annotate("Invocation.batch_size", domain="invocation"):
            if not self._is_batch:
                return None
            if self._batch_size is not None:
                return self._batch_size
            if self._results is None:
                # TODO(michalz): Try to get batch_size without full evaluation.
                with _EvalContext.get() as ctx:
                    self.run(ctx)
            return self._results[result_index].batch_size if self._is_batch else None

    def layout(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get layout without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].layout()

    def __getitem__(self, index):
        return InvocationResult(self, index)

    def __len__(self):
        if self._num_outputs is None:
            self._num_outputs = self._operator.infer_num_outputs(*self._inputs, **self._args)
        return self._num_outputs

    @property
    def is_batch(self):
        return self._is_batch

    def run(self, ctx: _EvalContext):
        if self._results is not None:
            return
        with nvtx.annotate("Invocation.run", domain="invocation"):
            if self._previous_invocation is not None:
                self._previous_invocation.run(ctx)
                self._previous_invocation = None
            cached = ctx.cached_results(self)
            if cached is not None:
                self._results = cached
            else:
                r = self._operator.run(
                    ctx,
                    *self._inputs,
                    batch_size=self._batch_size,
                    **self._args,
                )
                if isinstance(r, tuple) or isinstance(r, list):
                    self._results = list(r)
                else:
                    self._results = [r]
                if not self._is_batch:
                    self._results = [r[0] for r in self._results]
                self._results = tuple(self._results)
                ctx.cache_results(self, self._results)

    def values(self, ctx: _EvalContext):
        self.run(ctx)
        return self._results


class InvocationResult:
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

    def value(self, ctx: _EvalContext):
        return self._invocation.values(ctx)[self._index]

    @property
    def invocation(self):
        return self._invocation

    @property
    def index(self):
        return self._index
