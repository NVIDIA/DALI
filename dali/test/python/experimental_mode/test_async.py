# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import cuda_launch_host_func
from nose2.tools import params


@params(("cpu",), ("gpu",))
def test_simple_eager_execution(device):
    with ndd.EvalMode.eager:
        a = ndd.tensor([1, 2, 3], device=device)
        b = ndd.tensor([4, 5, 6], device=device)
        c = a + b

        np.testing.assert_array_equal(c.cpu(), [5, 7, 9])


@params(("cpu",), ("gpu",))
def test_chained_execution(device):
    with ndd.EvalMode.eager:
        a = ndd.tensor([2.0], device=device, dtype=ndd.float32)
        b = ndd.math.pow(a, 2.0)
        c = ndd.math.sqrt(b)

        np.testing.assert_array_equal(c.cpu(), [2.0])


@params(("cpu",), ("gpu",))
def test_independent_execution(device):
    with ndd.EvalMode.eager:
        t1 = ndd.tensor([10], device=device)
        t2 = ndd.tensor([20], device=device)

        op1 = t1 * ndd.tensor(2, device=device)
        op2 = t2 * ndd.tensor(3, device=device)

        np.testing.assert_array_equal(op1.cpu(), [20])
        np.testing.assert_array_equal(op2.cpu(), [60])


@params(("cpu",), ("gpu",))
def test_mixed_mode_switching(device):
    a = ndd.tensor([1], device=device)

    with ndd.EvalMode.eager:
        b = ndd.tensor([2], device=device)
        c = a + b

    np.testing.assert_array_equal(c.cpu(), [3])


@params(("cpu",), ("gpu",))
def test_no_deadlock(device):
    with ndd.EvalMode.eager:
        a = ndd.tensor([1, 2, 3, 4], device=device)
        # Here _ArithmeticGenericOp (ndd.math.sqrt) will trigger _TensorSubscript (a[-1])
        # Make sure that it doesn't deadlock
        b = ndd.math.sqrt(a[-1])

        assert b.item() == 2


def test_eager_parallelism():
    def wait_event(_):
        nonlocal started
        started = start_event.wait(1)

    start_event = threading.Event()
    cuda_stream = ndd.stream()
    cuda_launch_host_func(cuda_stream, wait_event)

    started = False
    with ndd.EvalMode.eager, ndd.EvalContext(cuda_stream=cuda_stream):
        a = ndd.tensor([1, 2, 3], device="gpu")
        b = ndd.tensor([4, 5, 6], device="gpu")
        # if the execution was not parallel, this would need to wait for the callback to be called
        # TODO(rtabet): find a way to reliably distinguish from lazy execution
        c = a + b
        start_event.set()

        c.evaluate()

        np.testing.assert_array_equal(c.cpu(), [5, 7, 9])
        assert started
