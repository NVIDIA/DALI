# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import scipy.stats as st
import math


def check_uniform_default(device='cpu', batch_size=32, shape=[1e5], val_range=None):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.uniform(device=device, range=val_range, shape=shape))
    pipe.build()
    outputs = pipe.run()
    val_range = (-1.0, 1.0) if val_range is None else val_range
    data_out = outputs[0].as_cpu() \
        if isinstance(outputs[0], TensorListGPU) else outputs[0]
    for i in range(batch_size):
        data = np.array(data_out[i])
        # Check that the data is within the default range
        assert (data >= val_range[0]).all() and \
               (data <= val_range[1]).all(), \
            "Value returned from the op is outside of requested range"

        h, b = np.histogram(data, bins=10)
        mean_h = np.mean(h)
        for hval in h:
            np.testing.assert_allclose(mean_h, hval, rtol=0.05)  # +/- 5%

        # normalize to 0-1 range
        data_kstest = (data - val_range[0]) / (val_range[1] - val_range[0])
        _, pv = st.kstest(rvs=data_kstest, cdf='uniform')
        assert pv > 0.05, f"data is not a uniform distribution (pv = {pv})"

def test_uniform_continuous():
    batch_size = 8
    shape = [100000]
    for device in ['cpu', 'gpu']:
        for val_range in [None, (200.0, 400.0)]:
            yield check_uniform_default, device, batch_size, shape, val_range

def check_uniform_discrete(device='cpu', batch_size=32, shape=[1e5], values=None):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.uniform(device=device, values=values, shape=shape))
    pipe.build()
    outputs = pipe.run()
    data_out = outputs[0].as_cpu() \
        if isinstance(outputs[0], TensorListGPU) else outputs[0]
    values_set = set(values)
    maxval = np.max(values)
    bins = np.concatenate([values, np.array([np.nextafter(maxval, maxval+1)])])
    bins.sort()
    for i in range(batch_size):
        data = np.array(data_out[i])
        for x in data:
            assert x in values_set
        h, _ = np.histogram(data, bins=bins)
        _, pv = st.chisquare(h)
        assert pv > 0.05, f"data is not a uniform distribution. pv = {pv}"

check_uniform_discrete('cpu', 32, shape=[10000], values=(0, 1, 2, 3, 4, 5, 6, 7))

def test_uniform_discrete():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    lo = -100
    hi = 100
    test_set = (hi - lo) * np.random.random_sample(10) + lo  # 10 elems from [-100, 100] range
    test_set = test_set.astype('float32')
    with pipe:
        pipe.set_outputs(dali.fn.uniform(values=test_set.tolist(), shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    test_set_max = np.max(test_set)
    bins = np.concatenate([test_set, np.array([np.nextafter(test_set_max, test_set_max+1)])])
    bins.sort()
    h, _ = np.histogram(possibly_uniform_distribution, bins=bins)[0]
    _, pv = st.chisquare(h)
    assert pv > 0.05, "`possibly_uniform_distribution` is not a uniform distribution"
    for val in possibly_uniform_distribution:
        assert val in test_set, \
            "Value returned from the op is outside of requested discrete set"
