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
# so it is better to store everything in one file and just call
# use_cupy` to switch between the default numpy and cupy

import numpy as np
import torch

import test_external_source_parallel_utils as utils
from nose_utils import raises


class ExtCallbackTorch(utils.ExtCallback):
    def __call__(self, sample_info):
        return torch.tensor(super().__call__(sample_info))


@raises(
    RuntimeError,
    "Error*starting Python worker threads for*parallel External Source*"
    "Cannot fork*CUDA has been initialized*"
    "*start_py_workers*fork*spawn*",
)
def test_pytorch_cuda_context():
    # Create a dummy torch CUDA tensor so we acquire CUDA context
    cuda0 = torch.device("cuda:0")
    _ = torch.ones([1, 1], dtype=torch.float32, device=cuda0)
    callback = utils.ExtCallback((4, 5), 10, np.int32)
    pipe = utils.create_pipe(
        callback, "cpu", 5, py_num_workers=6, py_start_method="fork", parallel=True
    )
    pipe.start_py_workers()


def test_pytorch():
    yield from utils.check_spawn_with_callback(ExtCallbackTorch)


class ExtCallbackTorchCuda(utils.ExtCallback):
    def __call__(self, sample_info):
        return torch.tensor(super().__call__(sample_info), device=torch.device("cuda:0"))


@raises(
    Exception,
    "Exception traceback received from worker thread*"
    "TypeError: Unsupported callback return type. GPU tensors*not supported*"
    "Got*PyTorch GPU tensor",
)
def test_pytorch_cuda():
    callback = ExtCallbackTorchCuda((4, 5), 10, np.int32)
    pipe = utils.create_pipe(
        callback, "cpu", 5, py_num_workers=6, py_start_method="spawn", parallel=True
    )
    utils.build_and_run_pipeline(pipe)
