# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def check_uniform_default(device="cpu", batch_size=32, shape=[1e5], val_range=None, niter=3):
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
                np.testing.assert_allclose(mean_h, hval, rtol=0.05)  # +/- 5%

            # normalize to 0-1 range
            data_kstest = (data - val_range[0]) / (val_range[1] - val_range[0])
            _, pv = st.kstest(rvs=data_kstest, cdf="uniform")
            pvs = pvs + [pv]
        assert np.mean(pvs) > 0.05, f"data is not a uniform distribution. pv = {np.mean(pvs)}"


def test_uniform_continuous():
    batch_size = 4
    shape = [100000]
    niter = 3
    for device in ["cpu", "gpu"]:
        for val_range in [None, (200.0, 400.0)]:
            yield check_uniform_default, device, batch_size, shape, val_range, niter


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


def test_uniform_continuous_next_after():
    batch_size = 4
    shape = [100000]
    niter = 3
    for device in ["cpu", "gpu"]:
        yield check_uniform_continuous_next_after, device, batch_size, shape, niter


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


def test_uniform_discrete():
    batch_size = 4
    shape = [10000]
    niter = 3
    for device in ["cpu", "gpu"]:
        for values in [(0, 1, 2, 3, 4, 5), (200, 400, 5000, 1)]:
            yield check_uniform_discrete, device, batch_size, shape, values, niter
