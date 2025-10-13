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

import nvidia.dali.experimental.dynamic as D


def test_eval_mode_context_manager():
    with D.EvalMode.eager:
        assert D.EvalMode.current() == D.EvalMode.eager
        with D.EvalMode.deferred:
            assert D.EvalMode.current() == D.EvalMode.deferred
        assert D.EvalMode.current() == D.EvalMode.eager


def test_eval_mode_comparison():
    assert D.EvalMode.eager.value > D.EvalMode.deferred.value
    assert D.EvalMode.sync_cpu.value > D.EvalMode.eager.value
    assert D.EvalMode.sync_full.value > D.EvalMode.sync_cpu.value
