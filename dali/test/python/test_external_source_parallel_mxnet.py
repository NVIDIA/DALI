# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# so it is better to store everything in one file and just call `use_cupy`
# to switch between the default numpy and cupy

import mxnet as mx
from nose_utils import raises, with_setup

from test_pool_utils import setup_function
from test_external_source_parallel_utils import (
    ExtCallback,
    check_spawn_with_callback,
    create_pipe,
    build_and_run_pipeline,
)
import numpy as np


class ExtCallbackMX(ExtCallback):
    def __call__(self, sample_info):
        a = super().__call__(sample_info)
        return mx.nd.array(a, dtype=a.dtype)


def test_mxnet():
    yield from check_spawn_with_callback(ExtCallbackMX)


class ExtCallbackMXCuda(ExtCallback):
    def __call__(self, sample_info):
        a = super().__call__(sample_info)
        return mx.nd.array(a, dtype=a.dtype, ctx=mx.gpu(0))


@raises(
    Exception,
    "Exception traceback received from worker thread*"
    "TypeError: Unsupported callback return type. GPU tensors*not supported*"
    "Got*MXNet GPU tensor.",
)
@with_setup(setup_function)
def test_mxnet_cuda():
    callback = ExtCallbackMXCuda((4, 5), 10, np.int32)
    pipe = create_pipe(callback, "cpu", 5, py_num_workers=6, py_start_method="spawn", parallel=True)
    build_and_run_pipeline(pipe)
