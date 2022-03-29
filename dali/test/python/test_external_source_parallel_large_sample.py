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

import numpy as np
from nose.tools import with_setup

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

from test_external_source_parallel_utils import setup_function, teardown_function, capture_processes


def large_sample_cb(sample_info):
    return np.full((512, 1024, 1024), sample_info.idx_in_epoch, dtype=np.int32)


@with_setup(setup_function, teardown_function)
def _test_large_sample(start_method):

    @pipeline_def
    def create_pipeline():
        large = fn.external_source(
            large_sample_cb, batch=False, parallel=True, prefetch_queue_depth=1)
        return large

    pipe = create_pipeline(batch_size=1, py_num_workers=2, py_start_method=start_method,
                           prefetch_queue_depth=1, num_threads=2, device_id=0)
    pipe.build()
    capture_processes(pipe._py_pool)
    for _ in range(8):
        pipe.run()


def test_large_sample():
    for start_method in ("fork", "spawn"):
        yield _test_large_sample, start_method
