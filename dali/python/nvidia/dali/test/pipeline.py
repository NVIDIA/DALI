# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali


@dali.pipeline_def
def pipe():
    return 42


@dali.pipeline_def()
def pipe_unconf():
    return 42


@dali.pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
def pipe_conf():
    return 42


def test_is_pipeline_def():
    assert getattr(pipe, 'is_pipeline_def', False)
    assert getattr(pipe_unconf, 'is_pipeline_def', False)
    assert getattr(pipe_conf, 'is_pipeline_def', False)
