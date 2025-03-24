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
from ._tensor import Tensor, TensorSlice
from ._eval_context import _EvalContext
from ._type import DataType


class Expression:
    def __init__(
        self,
        operator_instance,
        call_id,
        inputs=[],
        args={},
        is_batch: bool = False,
        batch_size: Optional[int] = None,
    ):
        self._operator = operator_instance
        self._call_id = call_id
        self._inputs = inputs
        self._args = args
        self._is_batch = is_batch
        self._batch_size = batch_size
        self._result = None

    @property
    def ndim(self) -> int:
        if self._result is None:
            # TODO(michalz): Try to get ndim without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._result.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._result is None:
            # TODO(michalz): Try to get shape without full evaluation.
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._result.shape

    @property
    def dtype(self) -> DataType:
        if self._result is None:
            with _EvalContext.get() as ctx:
                self.run(ctx)
        return self._result.dtype

    def run(self, ctx: _EvalContext):
        if self._result is None:
            cached = ctx.cached_result(self)
            if cached is not None:
                self._result = cached
            else:
                self._result = self._operator.run(*self._inputs, **self._args)
                ctx.cache_result(self, self._result)

        return self._result
