# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# it is enough to just import all functions from test_internals_operator_external_source
# nose will query for the methods available and will run them
# the test_internals_operator_external_source is 99% the same for cupy and numpy tests
# so it is better to store everything in one file and just call `use_cupy` to switch between the default numpy and cupy
from test_external_source_impl import *
from test_utils import check_output_pattern
use_cupy()

# extra tests, GPU-specific
import cupy as cp
import os

def test_external_source_with_iter_cupy_stream():
    with cp.cuda.Stream(non_blocking=True):
        for attempt in range(10):
            pipe = Pipeline(1, 3, 0)

            pipe.set_outputs(fn.external_source(lambda i: [cp.array([attempt * 100 + i * 10 + 1.5], dtype=cp.float32)]))
            pipe.build()

            for i in range(10):
                check_output(pipe.run(), [np.array([attempt * 100 + i * 10 + 1.5], dtype=np.float32)])

def test_external_source_mixed_contiguous():
    batch_size = 2
    iterations = 4
    def generator(i):
        if i % 2:
            return cp.array([100 + i * 10 + 1.5] * batch_size, dtype=cp.float32)
        else:
            return batch_size * [cp.array([100 + i * 10 + 1.5], dtype=cp.float32)]

    pipe = Pipeline(batch_size, 3, 0)

    pipe.set_outputs(fn.external_source(device="gpu", source=generator, no_copy=True))
    pipe.build()

    pattern = "ExternalSource operator should not mix contiguous and noncontiguous inputs. " \
              "In such a case the internal memory used to gather data in a contiguous chunk of " \
              "memory would be trashed."
    with check_output_pattern(pattern):
        for _ in range(iterations):
            pipe.run()
