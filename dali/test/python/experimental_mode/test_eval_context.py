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
from nose_utils import SkipTest, assert_raises

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
    with dali2.EvalContext(device_id=0) as ctx0:
        assert dali2.EvalContext.current is ctx0
        assert dali2.EvalContext.current.device_id == 0
        with dali2.EvalContext(device_id=1) as ctx1:
            assert dali2.EvalContext.current is not ctx0
            assert dali2.EvalContext.current is ctx1
            assert dali2.EvalContext.current.device_id == 1
        assert dali2.EvalContext.current is ctx0
        assert dali2.EvalContext.current.device_id == 0
    assert dali2.EvalContext.current is dali2.EvalContext.default()
    assert dali2.EvalContext.current.device_id is None

def test_eval_context_explicit_stream():
    with dali2.EvalContext.get() as ctx:
        s = ctx.cuda_stream
        s2 = _backend.Stream()
        with dali2.EvalContext(cuda_stream=s2) as ctx2:
            assert dali2.EvalContext.current.cuda_stream is s2

def test_eval_context_multi_gpu():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    with dali2.EvalContext(device_id=0):
        assert dali2.EvalContext.current.device_id == 0
        with dali2.EvalContext(device_id=1):
            assert dali2.EvalContext.current.device_id == 1
        assert dali2.EvalContext.current.device_id == 0
    assert dali2.EvalContext.current.device_id is None

class PseudoInvocation:
    def __init__(self, value):
        self.value = value
        self.run_count = 0


    def run(self, ctx):
        assert isinstance(ctx, dali2.EvalContext)
        self.run_count += 1
        return self.value

def test_eval_context_cached_results():
    with dali2.EvalContext.get() as ctx:
        inv = PseudoInvocation(1)
        assert ctx.cached_results(inv) is None
        assert inv.run_count == 0
        ctx.cache_results(inv, 2)
        assert ctx.cached_results(inv) == 2
        assert ctx.cached_results(inv) == 2  # should be cached
        assert ctx.cached_results(PseudoInvocation(3)) is None
        assert ctx.cached_results(PseudoInvocation(3)) is None  # should be cached