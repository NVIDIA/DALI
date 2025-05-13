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
from ._eval_context import EvalContext as _EvalContext
from ._type import DType


class Invocation:
    def __init__(
        self,
        operator_instance,
        call_id,
        inputs=[],
        args={},
        is_batch: bool = False,
    ):
        self._operator = operator_instance
        self._call_id = call_id
        self._inputs = inputs
        self._args = args
        self._is_batch = is_batch
        self._results = None

    def ndim(self, result_index: int) -> int:
        if self._results is None:
            # TODO(michalz): Try to get ndim without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].ndim

    def shape(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get shape without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].shape

    def dtype(self, result_index: int) -> DType:
        if self._results is None:
            # TODO(michalz): Try to get dtype without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].dtype

    def batch_size(self, result_index: int):
        if self._results is None:
            # TODO(michalz): Try to get batch_size without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._results[result_index].batch_size if self._is_batch else None

    def __getitem__(self, index):
        return InvocationResult(self, index)

    @property
    def is_batch(self):
        return self._is_batch


    def values(self, ctx: _EvalContext):
        if self._results is None:
            cached = ctx.cached_result(self)
            if cached is not None:
                self._results = cached
            else:
                r = self._operator.run(*self._inputs, **self._args)
                if isinstance(r, tuple) or isinstance(r, list):
                    self._results = tuple(r)
                else:
                    self._results = (r,)
                ctx.cache_result(self, self._result)

        return self._results

class InvocationResult:
    def __init__(self, invocation, index: int):
        self._invocation = invocation
        self._index = index

    def ndim(self) -> int:
        return self._invocation.ndim(self._index)

    def shape(self):
        return self._invocation.shape(self._index)

    def dtype(self) -> DType:
        return self._invocation.dtype(self._index)

    def batch_size(self):
        return self._invocation.batch_size(self._index)

    def value(self):
        return self._invocation.values(self._index)
