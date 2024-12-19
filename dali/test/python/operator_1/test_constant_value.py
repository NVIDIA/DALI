# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import pipeline_def, fn


def run(op):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe0():
        return op

    p = pipe0()
    return np.array(p.run()[0][0])


def test_zeros():
    sh = (2, 3)
    np.testing.assert_array_equal(run(fn.zeros(shape=sh)), np.zeros(shape=sh))


def test_zeros_like():
    sh = (2, 3)
    arr = np.ones(sh)
    np.testing.assert_array_almost_equal(run(fn.zeros_like(arr)), np.zeros_like(arr))


def test_ones():
    sh = (2, 3)
    np.testing.assert_array_almost_equal(run(fn.ones(shape=sh)), np.ones(shape=sh))


def test_ones_like():
    sh = (2, 3)
    arr = np.ones(sh)
    np.testing.assert_array_almost_equal(run(fn.ones_like(arr)), np.ones_like(arr))


def test_full():
    sh = (2, 3, 4)
    fill_value_sh = (3, 4)
    fill_value_arr = np.random.uniform(size=fill_value_sh)
    np.testing.assert_array_almost_equal(
        run(fn.full(fill_value_arr, shape=sh)), np.full(sh, fill_value_arr)
    )


def test_full_like():
    sh = (2, 3, 4)
    fill_value_sh = (3, 4)
    arr = np.random.uniform(size=sh)
    fill_value_arr = np.random.uniform(size=fill_value_sh)
    np.testing.assert_array_almost_equal(
        run(fn.full_like(arr, fill_value_arr)), np.full_like(arr, fill_value_arr)
    )
