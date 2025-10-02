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

import nvidia.dali.experimental.dali2 as dali2
import nvidia.dali.backend as _backend
from nose_utils import SkipTest, attr


def test_eval_context_get():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")
    ctx = dali2.EvalContext.get()
    assert ctx is not None
    assert dali2.EvalContext.get() is ctx  # get() should not create another context
    assert ctx.device_id == _backend.GetCUDACurrentDevice()
    s = ctx.cuda_stream
    assert s is not None
    assert ctx.cuda_stream is s
    assert ctx is dali2.EvalContext.default()
    assert dali2.EvalContext.get().cuda_stream == s  # get() should not recreate the stream


def test_eval_context_context_manager():
    other_device_id = 1 if _backend.GetCUDADeviceCount() > 1 else 0
    if other_device_id == 0:
        print("Warning: Only 1 GPU detected, weak test")
    with dali2.EvalContext(device_id=0) as ctx0:
        assert dali2.EvalContext.current() is ctx0
        assert dali2.EvalContext.current().device_id == 0
        with dali2.EvalContext(device_id=other_device_id) as ctx1:
            assert dali2.EvalContext.current() is not ctx0
            assert dali2.EvalContext.current() is ctx1
            assert dali2.EvalContext.current().device_id == other_device_id
        assert dali2.EvalContext.current() is ctx0
        assert dali2.EvalContext.current().device_id == 0
    assert dali2.EvalContext.current() is dali2.EvalContext.default()


def test_eval_context_explicit_stream():
    with dali2.EvalContext.get() as ctx:
        s = ctx.cuda_stream
        s2 = _backend.Stream(0)
        with dali2.EvalContext(cuda_stream=s2) as ctx2:
            assert dali2.EvalContext.current() is ctx2
            assert dali2.EvalContext.current().cuda_stream is s2
        assert dali2.EvalContext.current().cuda_stream is s


@attr("multi_gpu")
def test_eval_context_multi_gpu():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    assert _backend.GetCUDACurrentDevice() == 0, "Invalid initial device id"
    with dali2.EvalContext(device_id=0) as ctx0:
        assert dali2.EvalContext.current() is ctx0
        with dali2.EvalContext(device_id=1) as ctx1:
            assert dali2.EvalContext.current() is ctx1
            assert dali2.EvalContext.current().device_id == 1
        assert dali2.EvalContext.current() is ctx0
        assert dali2.EvalContext.current().device_id == 0
    assert dali2.EvalContext.current() is dali2.EvalContext.default()


class PseudoInvocation:
    def __init__(self, value, run_count_container=None):
        self.value = value
        self.run_count = 0
        self.run_count_container = run_count_container

    def run(self, ctx):
        assert isinstance(ctx, dali2.EvalContext)
        self.run_count += 1
        if self.run_count_container is not None:
            self.run_count_container.run_count += 1
        ctx.cache_results(self, self.value)
        return self.value


def test_eval_context_cached_results():
    with dali2.EvalContext.get() as ctx:
        inv = PseudoInvocation(42)
        assert ctx.cached_results(inv) is None
        inv.run(ctx)
        assert inv.run_count == 1
        assert ctx.cached_results(inv) == 42


def test_eval_context_evaluate_all():
    with dali2.EvalContext() as ctx:
        inv = PseudoInvocation(4321)
        ctx._add_invocation(inv, weak=False)
        assert inv.run_count == 0
    assert inv.run_count == 1


def test_eval_evaluate_all_skip_cached():
    with dali2.EvalContext() as ctx:
        inv = PseudoInvocation(42)
        assert ctx.cached_results(inv) is None
        assert inv.run_count == 0
        ctx.cache_results(inv, 123)
        assert ctx.cached_results(inv) == 123
        ctx._add_invocation(inv)
        ctx.evaluate_all()
        assert inv.run_count == 0  # cached
        ctx._cached_results = {}
        ctx._add_invocation(inv)
        ctx.evaluate_all()
        assert inv.run_count == 1
        assert ctx.cached_results(inv) == 42
        ctx.evaluate_all()
        assert inv.run_count == 1


def test_eval_context_evaluate_all_weakref():
    run_count_container = PseudoInvocation(0)
    with dali2.EvalContext() as ctx:
        inv = PseudoInvocation(1057, run_count_container)  # lost
        ctx._add_invocation(inv, weak=True)
        del inv
    assert run_count_container.run_count == 0
