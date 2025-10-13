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


def test_eval_mode_context_manager():
    with ndd.EvalMode.eager:
        assert ndd.EvalMode.current() == ndd.EvalMode.eager
        with ndd.EvalMode.deferred:
            assert ndd.EvalMode.current() == ndd.EvalMode.deferred
        assert ndd.EvalMode.current() == ndd.EvalMode.eager


def test_eval_mode_comparison():
    assert ndd.EvalMode.eager.value > ndd.EvalMode.deferred.value
    assert ndd.EvalMode.sync_cpu.value > ndd.EvalMode.eager.value
    assert ndd.EvalMode.sync_full.value > ndd.EvalMode.sync_cpu.value
