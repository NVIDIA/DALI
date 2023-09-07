# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.jax.iterator import DALIGenericIterator

from clu.data.dataset_iterator import ArraySpec, Element, ElementSpec
from clu import asynclib


def jax_array_to_array_spec(jax_array):
    return ArraySpec(
        shape=jax_array.shape,
        dtype=jax_array.dtype)


class DALIGenericPeekableIterator(DALIGenericIterator):
    """DALI iterator for JAX with peek functionality. Compatible with Google CLU PeekableIterator.

    """
    def __init__(
            self,
            pipelines,
            output_map,
            size=-1,
            reader_name=None,
            auto_reset=False,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=True,
            sharding=None):
        super().__init__(
            pipelines,
            output_map,
            size,
            reader_name,
            auto_reset,
            last_batch_padded,
            last_batch_policy,
            prepare_first_batch,
            sharding)
        self._mutex = threading.Lock()
        self._pool = None
        self._peek_future = None
        self._element_spec = None
        
        # Set element spec
        peeked_output = self.peek()
        self._element_spec = {
            output_name: jax_array_to_array_spec(peeked_output[output_name])
            for output_name in self._output_categories
        }

    def __next__(self):
        with self._mutex:
            if self._peek is None:
                return self.next_impl()
        peek = self._peek
        self._peek = None
        return peek

    def peek_async(self):
        with self._mutex:
            if self._peek_future is None:
                if self._pool is None:
                    self._pool = asynclib.Pool(max_workers=1)
                self._peek_future = self._pool(self.peek)()
        return self._peek_future

    @property
    def element_spec(self) -> ElementSpec:
        return self._element_spec
