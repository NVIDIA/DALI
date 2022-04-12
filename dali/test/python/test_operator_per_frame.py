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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali import types

from test_utils import check_batch
from nose_utils import raises

batch_sizes = [5, 256, 128, 7]
max_batch_size = max(batch_sizes)


def input_batch(num_dim):
    rng = np.random.default_rng(42)
    for batch_size in batch_sizes:
        yield [rng.random(rng.integers(low=0, high=50, size=num_dim)) for _ in range(batch_size)]


def run_pipeline(num_dim, replace=False, layout=None):

    @pipeline_def
    def pipeline():
        arg = fn.external_source(input_batch(num_dim), layout=layout)
        return fn.per_frame(arg, replace=replace)

    pipe = pipeline(num_threads=4, batch_size=max_batch_size,
                    device_id=types.CPU_ONLY_DEVICE_ID)
    pipe.build()
    expected_layout = "F" + "*" * (num_dim - 1) if layout is None else "F" + layout[1:]
    for baseline in input_batch(num_dim):
        (out,) = pipe.run()
        check_batch(out, baseline, len(baseline), expected_layout=expected_layout)


def test_set_layout():
    for num_dim in (1, 2, 3):
        yield run_pipeline, num_dim


def test_replace_layout():
    for num_dim in (1, 2, 3):
        yield run_pipeline, num_dim, True, "XYZ"[:num_dim]


def test_verify_layout():
    for num_dim in (1, 2, 3):
        yield run_pipeline, num_dim, False, "FYZ"[:num_dim]


@raises(RuntimeError, "Cannot mark zero-dimensional input as a sequence")
def test_zero_dim_not_allowed():
    run_pipeline(num_dim=0)


@raises(RuntimeError, " Per-frame argument input must be a sequence. The input layout should start with 'F'")
def _test_not_a_sequence_layout(num_dim, layout):
    run_pipeline(num_dim=num_dim, layout=layout)


def test_not_a_sequence_layout():
    for num_dim in (1, 2, 3):
        yield _test_not_a_sequence_layout, num_dim, "XYZ"[:num_dim]
