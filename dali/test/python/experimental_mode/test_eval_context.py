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
from nose_utils import SkipTest, attr
import numpy as np
import os
from nose2.tools import cartesian_params
from test_utils import get_dali_extra_path


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
    result = result.evaluate()
    result_cpu = result.to_device(ndd.Device("cpu"))
    result_cpu = result_cpu.evaluate()
    assert result_cpu.shape == (1,)
    assert result_cpu.dtype == ndd.float32
    assert result_cpu.device.device_type == "cpu"
    assert result_cpu.device.device_id == 0
    np.testing.assert_array_equal(result_cpu, np.array([5.0], dtype=np.float32))


@cartesian_params(
    [
        None,
    ]
    + list(range(_backend.GetCUDADeviceCount())),
)
def test_eval_context_evaluate_gpu_expr(device_id):
    """
    Test that operations executed on the device context used to create the invocation, regardless
    of the current eval context.
    """
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    if device_id is None:
        result = _gpu_expr()
    else:
        with ndd.Device(f"gpu:{device_id}"):
            result = _gpu_expr()
    actual_device_id = device_id if device_id is not None else 0
    # It doesn't matter if the current eval context is different from the one used to create
    # the invocation, since the eval context is captured when the invocation is created.
    for eval_device_id in range(_backend.GetCUDADeviceCount()):
        with ndd.EvalContext(device_id=eval_device_id):
            _validate_gpu_expr_result(result, actual_device_id)


def test_default_device_conversion_to_cpu():
    """
    Test that evaluating a GPU tensor converted to CPU doesn't trigger any device mismatch errors.
    """
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    cpu_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gpu_tensor = ndd.tensor(cpu_arr, device="gpu")
    cpu_tensor = gpu_tensor.cpu()
    cpu_tensor_result = cpu_tensor.evaluate()
    np.testing.assert_array_equal(cpu_tensor_result, cpu_arr)


@cartesian_params(
    [
        None,
    ]
    + list(range(_backend.GetCUDADeviceCount())),
)
def test_device_match_mixed_operator(device_id):
    """
    Test that mixed operators (such as decoders.image) correctly honor the active device context.
    """
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")

    image_path = os.path.join(
        get_dali_extra_path(), "db", "single", "jpeg", "100", "swan-3584559_640.jpg"
    )
    with open(image_path, "rb") as f:
        # Use .copy() to create a writable array so that it can be passed as a Tensor through dlpack
        encoded_data = np.frombuffer(f.read(), dtype=np.uint8).copy()

    if device_id is None:
        decoded_gpu = ndd.decoders.image(encoded_data, device="mixed")
    else:
        with ndd.Device(f"gpu:{device_id}"):
            decoded_gpu = ndd.decoders.image(encoded_data, device="mixed")

    eval_device_id = device_id if device_id is not None else 0
    with ndd.EvalContext(device_id=eval_device_id):
        assert decoded_gpu.device.device_type == "gpu"
        assert decoded_gpu.device.device_id == eval_device_id
        output = decoded_gpu.evaluate()
        # Image dimensions for swan-3584559_640.jpg
        assert output.ndim == 3
        assert output.shape == (408, 640, 3)
        assert output.dtype == ndd.uint8
