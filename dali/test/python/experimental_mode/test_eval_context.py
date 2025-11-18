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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.backend as _backend
from nose_utils import SkipTest, attr, assert_raises
import numpy as np
import cupy as cp


def test_eval_context_get():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")
    ctx = ndd.EvalContext.current()
    assert ctx is not None
    assert ndd.EvalContext.current() is ctx  # get() should not create another context
    assert ctx.device_id == _backend.GetCUDACurrentDevice()
    s = ctx.cuda_stream
    assert s is not None
    assert ctx.cuda_stream is s
    assert ctx is ndd.EvalContext.default()
    assert ndd.EvalContext.current().cuda_stream == s  # get() should not recreate the stream


def test_eval_context_context_manager():
    other_device_id = 1 if _backend.GetCUDADeviceCount() > 1 else 0
    if other_device_id == 0:
        print("Warning: Only 1 GPU detected, weak test")
    with ndd.EvalContext(device_id=0) as ctx0:
        assert ndd.EvalContext.current() is ctx0
        assert ndd.EvalContext.current().device_id == 0
        with ndd.EvalContext(device_id=other_device_id) as ctx1:
            assert ndd.EvalContext.current() is not ctx0
            assert ndd.EvalContext.current() is ctx1
            assert ndd.EvalContext.current().device_id == other_device_id
        assert ndd.EvalContext.current() is ctx0
        assert ndd.EvalContext.current().device_id == 0
    assert ndd.EvalContext.current() is ndd.EvalContext.default()


def test_eval_context_explicit_stream():
    with ndd.EvalContext.current() as ctx:
        s = ctx.cuda_stream
        s2 = _backend.Stream(0)
        with ndd.EvalContext(cuda_stream=s2) as ctx2:
            assert ndd.EvalContext.current() is ctx2
            assert ndd.EvalContext.current().cuda_stream is s2
        assert ndd.EvalContext.current().cuda_stream is s


@attr("multi_gpu")
def test_eval_context_multi_gpu():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    assert _backend.GetCUDACurrentDevice() == 0, "Invalid initial device id"
    with ndd.EvalContext(device_id=0) as ctx0:
        assert ndd.EvalContext.current() is ctx0
        with ndd.EvalContext(device_id=1) as ctx1:
            assert ndd.EvalContext.current() is ctx1
            assert ndd.EvalContext.current().device_id == 1
        assert ndd.EvalContext.current() is ctx0
        assert ndd.EvalContext.current().device_id == 0
    assert ndd.EvalContext.current() is ndd.EvalContext.default()


class PseudoInvocation:
    def __init__(self, value, run_count_container=None):
        self.value = value
        self.run_count = 0
        self.run_count_container = run_count_container

    def run(self, ctx):
        assert isinstance(ctx, ndd.EvalContext)
        self.run_count += 1
        if self.run_count_container is not None:
            self.run_count_container.run_count += 1
        ctx.cache_results(self, self.value)
        return self.value


def test_eval_context_evaluate_all():
    with ndd.EvalContext() as ctx:
        inv = PseudoInvocation(4321)
        ctx._add_invocation(inv, weak=False)
        assert inv.run_count == 0
    assert inv.run_count == 1


# TODO(michalz): Result caching disabled due to a bug. It needs a redesign.

# def test_eval_context_cached_results():
#     with ndd.EvalContext.current() as ctx:
#         inv = PseudoInvocation(42)
#         assert ctx.cached_results(inv) is None
#         inv.run(ctx)
#         assert inv.run_count == 1
#         assert ctx.cached_results(inv) == 42


# def test_eval_evaluate_all_skip_cached():
#     with ndd.EvalContext() as ctx:
#         inv = PseudoInvocation(42)
#         assert ctx.cached_results(inv) is None
#         assert inv.run_count == 0
#         ctx.cache_results(inv, 123)
#         assert ctx.cached_results(inv) == 123
#         ctx._add_invocation(inv)
#         ctx.evaluate_all()
#         assert inv.run_count == 0  # cached
#         ctx._cached_results = {}
#         ctx._add_invocation(inv)
#         ctx.evaluate_all()
#         assert inv.run_count == 1
#         assert ctx.cached_results(inv) == 42
#         ctx.evaluate_all()
#         assert inv.run_count == 1


def test_eval_context_evaluate_all_weakref():
    run_count_container = PseudoInvocation(0)
    with ndd.EvalContext() as ctx:
        inv = PseudoInvocation(1057, run_count_container)  # lost
        ctx._add_invocation(inv, weak=True)
        del inv
    assert run_count_container.run_count == 0


def _gpu_expr():
    a = ndd.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device="gpu")
    b = ndd.tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32), device="gpu")
    return ndd.slice(a + b, end=[1], axes=[0])


def _validate_gpu_expr_result(result, expected_device_id):
    assert result.device.device_type == "gpu"
    assert result.device.device_id == expected_device_id
    assert result.shape == (1,)
    output_cp = cp.from_dlpack(result._storage)
    expected_cp = cp.array([5.0], dtype=cp.float32)
    assert cp.allclose(output_cp, expected_cp)


@attr("multi_gpu")
def test_device_mismatch_explicit_device():
    """
    Test that we properly detect when an invocation created with one device
    is evaluated with a different device context.
    """
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")

    # Create an invocation in GPU 1 context
    with ndd.Device("gpu:1"):
        result = _gpu_expr()

    # Try to evaluate with GPU 0 context - this should raise a RuntimeError
    with assert_raises(RuntimeError, glob="*Device mismatch*gpu:1*gpu:0*"):
        with ndd.EvalContext(device_id=0):
            result.evaluate()


@attr("multi_gpu")
def test_device_mismatch_default_device():
    """
    Test that we properly detect when an invocation created with one device
    is evaluated with a different device context.
    """
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")

    result = _gpu_expr()

    # Try to evaluate with GPU 0 context - this should raise a RuntimeError
    with assert_raises(RuntimeError, glob="*Device mismatch*gpu:0*gpu:1*"):
        with ndd.EvalContext(device_id=1):
            result.evaluate()


def test_device_match_explicit_device():
    """
    Test that operations work fine when device contexts match the default device.
    """
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    for device_id in range(_backend.GetCUDADeviceCount()):
        with ndd.Device(f"gpu:{device_id}"):
            result = _gpu_expr()
        with ndd.EvalContext(device_id=device_id):
            output = result.evaluate()
            _validate_gpu_expr_result(output, device_id)


def test_device_match_default_device():
    """
    Test that slice operator works correctly when device contexts match.
    """
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    # Create and evaluate slice operation with matching device context
    with ndd.Device("gpu:0"):
        sliced = _gpu_expr()

    with ndd.EvalContext(device_id=0):
        output = sliced.evaluate()
        _validate_gpu_expr_result(output, 0)
