# Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.pipeline import Pipeline
import numpy as np
import scipy.stats as st


def check_uniform_default(
    device="cpu", batch_size=32, shape=[1e5], val_range=None, niter=3, bin_tol=0.05
):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.uniform(device=device, range=val_range, shape=shape))
    for _ in range(niter):
        (data_out,) = tuple(out.as_cpu() for out in pipe.run())
        val_range = (-1.0, 1.0) if val_range is None else val_range
        pvs = []
        for i in range(batch_size):
            data = np.array(data_out[i])
            # Check that the data is within the default range
            assert (data >= val_range[0]).all() and (
                data <= val_range[1]
            ).all(), "Value returned from the op is outside of requested range"

            h, b = np.histogram(data, bins=10)
            mean_h = np.mean(h)
            for hval in h:
                np.testing.assert_allclose(mean_h, hval, rtol=bin_tol)

            # normalize to 0-1 range
            data_kstest = (data - val_range[0]) / (val_range[1] - val_range[0])
            _, pv = st.kstest(rvs=data_kstest, cdf="uniform")
            pvs = pvs + [pv]
        assert np.mean(pvs) > 0.05, f"data is not a uniform distribution. pv = {np.mean(pvs)}"


from nose2.tools import params


_uniform_continuous_test_cases = [
    (device, 4, [100000], val_range, 3)
    for device in ["cpu", "gpu"]
    for val_range in [None, (200.0, 400.0)]
]


@params(*_uniform_continuous_test_cases)
def test_uniform_continuous(device, batch_size, shape, val_range, niter):
    check_uniform_default(device, batch_size, shape, val_range, niter)


@params("cpu", "gpu")
def test_uniform_large_batch(device):
    check_uniform_default(device, 2000, [2000], [-1, 1], 2, 0.45)


def check_uniform_continuous_next_after(device="cpu", batch_size=32, shape=[1e5], niter=3):
    batch_size = 4
    shape = [100000]
    val_range = [np.float32(10.0), np.nextafter(np.float32(10.0), np.float32(11.0))]

    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.uniform(device=device, range=val_range, shape=shape))
    for _ in range(niter):
        (data_out,) = tuple(out.as_cpu() for out in pipe.run())
        for i in range(batch_size):
            data = np.array(data_out[i])
            assert (val_range[0] == data).all(), f"{data} is outside of requested range"


@params("cpu", "gpu")
def test_uniform_continuous_next_after(device):
    check_uniform_continuous_next_after(device, 4, [100000], 3)


def check_uniform_discrete(device="cpu", batch_size=32, shape=[1e5], values=None, niter=10):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.uniform(device=device, values=values, shape=shape))
    for _ in range(niter):
        (data_out,) = tuple(out.as_cpu() for out in pipe.run())
        values_set = set(values)
        maxval = np.max(values)
        bins = np.concatenate([values, np.array([np.nextafter(maxval, maxval + 1)])])
        bins.sort()
        pvs = []
        for i in range(batch_size):
            data = np.array(data_out[i])
            for x in data:
                assert x in values_set
            h, _ = np.histogram(data, bins=bins)
            _, pv = st.chisquare(h)
            pvs = pvs + [pv]
        assert np.mean(pvs) > 0.05, f"data is not a uniform distribution. pv = {np.mean(pvs)}"


_uniform_discrete_test_cases = [
    (device, 4, [10000], values, 3)
    for device in ["cpu", "gpu"]
    for values in [(0, 1, 2, 3, 4, 5), (200, 400, 5000, 1)]
]


@params(*_uniform_discrete_test_cases)
def test_uniform_discrete(device, batch_size, shape, values, niter):
    check_uniform_discrete(device, batch_size, shape, values, niter)


def check_uniform_with_random_state(device, batch_size, shape, niter):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        state1 = dali.fn.external_source(
            [
                [np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)] * batch_size,
                [np.array([8, 9, 10, 11, 12, 13, 14], dtype=np.uint32)] * batch_size,
            ],
            batch=True,
            cycle=True,
        )
        state2 = dali.fn.external_source(
            [
                [np.array([100, 200, 300, 400, 500, 600, 700], dtype=np.uint32)] * batch_size,
                [np.array([800, 900, 1000, 1100, 1200, 1300, 1400], dtype=np.uint32)] * batch_size,
                [np.array([1500, 1600, 1700, 1800, 1900, 2000, 2100], dtype=np.uint32)]
                * batch_size,
            ],
            batch=True,
            cycle=True,
        )
        u1_a = dali.fn.random.uniform(device=device, shape=shape, _random_state=state1)
        u1_b = dali.fn.random.uniform(device=device, shape=shape, _random_state=state1)
        u2_a = dali.fn.random.uniform(device=device, shape=shape, _random_state=state2)
        u2_b = dali.fn.random.uniform(device=device, shape=shape, _random_state=state2)
        pipe.set_outputs(u1_a, u1_b, u2_a, u2_b)

    outputs = []
    for iter in range(niter):
        data_out = tuple(out.as_cpu() for out in pipe.run())
        outputs.append(data_out)
        for i in range(batch_size):
            data1_a = np.array(data_out[0][i])
            data1_b = np.array(data_out[1][i])
            data2_a = np.array(data_out[2][i])
            data2_b = np.array(data_out[3][i])
            assert np.array_equal(data1_a, data1_b), f"{data1_a} should be equal to {data1_b}"
            assert np.array_equal(data2_a, data2_b), f"{data2_a} should be equal to {data2_b}"
            assert not np.array_equal(
                data1_a, data2_a
            ), f"{data1_a} should be different from {data2_a}"
            assert not np.array_equal(
                data1_b, data2_b
            ), f"{data1_b} should be different from {data2_b}"
            if iter == 2:
                # random state 1 has a period of 2 batches
                assert np.array_equal(data1_a, outputs[0][0][i])
                assert np.array_equal(data1_b, outputs[0][1][i])
            if iter == 3:
                # random state 2 has a period of 3 batches
                assert np.array_equal(data2_a, outputs[0][2][i])
                assert np.array_equal(data2_b, outputs[0][3][i])


@params("cpu", "gpu")
def test_uniform_random_state(device):
    check_uniform_with_random_state(device, 4, [1000], 4)
