# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.backend as _b
import numpy as np
from nose_utils import attr, SkipTest


@attr("cpu_only")
def test_eval_context_current():
    if _b.GetCUDADeviceCount() > 0:
        raise SkipTest("This test doesn't work with GPU present")
    ctx = ndd.EvalContext.current()
    assert ctx.device_id is None


@attr("cpu_only")
def test_construct_tensor():
    t = ndd.tensor([1, 2, 3])
    assert np.array_equal(t, np.array([1, 2, 3]))
    t = np.array([1, 2, 3], dtype=np.float32)
    assert np.array_equal(t, np.array([1, 2, 3], dtype=np.float32))


@attr("cpu_only")
def test_construct_batch():
    b = ndd.batch([[1, 2, 3], [4, 5]])
    assert np.array_equal(b.tensors[0], np.array([1, 2, 3]))
    assert np.array_equal(b.tensors[1], np.array([4, 5]))


@attr("cpu_only")
def test_copy_tensor():
    t = ndd.tensor([1, 2, 3])
    t2 = ndd.tensor(t)
    assert np.array_equal(t2, np.array([1, 2, 3], dtype=np.float32))


@attr("cpu_only")
def test_copy_batch():
    b1 = ndd.batch([[1, 2, 3], [4, 5]])
    b2 = ndd.batch(b1)
    assert np.array_equal(b2.tensors[0], np.array([1, 2, 3]))
    assert np.array_equal(b2.tensors[1], np.array([4, 5]))


@attr("cpu_only")
def test_add_tensors():
    t1 = ndd.tensor([1, 2, 3])
    t2 = ndd.tensor([4.0, 5.0, 6.0])
    t3 = t1 + t2
    assert np.array_equal(t3, np.array([5, 7, 9], dtype=np.float32))


@attr("cpu_only")
def test_add_batches():
    b1 = ndd.batch([[1, 2, 3], [4, 5]])
    b2 = ndd.batch([[5], [42]])
    b3 = b1 + b2
    assert np.array_equal(b3.tensors[0], np.array([6, 7, 8]))
    assert np.array_equal(b3.tensors[1], np.array([46, 47]))
