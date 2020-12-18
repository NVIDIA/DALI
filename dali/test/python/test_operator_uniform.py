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


def check_uniform_default(device='cpu', batch_size=8, shape=[1e5], val_range=None):
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
        print("sample ", i)
        print("histogram ", h)
        print("min/max ", np.min(data), np.max(data))

        # normalize to 0-1 range
        data_kstest = (data - val_range[0]) / (val_range[1] - val_range[0])
        _, pv = st.kstest(rvs=data_kstest, cdf='uniform')
        print("pv: ", pv)
        # assert pv > 0.05, f"data is not a uniform distribution (pv = {pv})"

check_uniform_default('gpu', val_range=(200.0, 300.0))

def test_uniform_default():
    for device in ['cpu', 'gpu']:
        yield check_uniform_default, device

def test_uniform_continuous():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    lo = -100
    hi = 100
    test_range = (hi - lo) * np.random.random_sample(2) + lo  # 2 elems from [-100, 100] range
    test_range.sort()
    test_range = test_range.astype('float32')
    with pipe:
        pipe.set_outputs(dali.fn.uniform(range=test_range.tolist(), shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    # normalize to 0-1 range
    possibly_uniform_distribution_kstest = (possibly_uniform_distribution - test_range[0]) / (test_range[1] - test_range[0])
    _, pv = st.kstest(rvs=possibly_uniform_distribution_kstest, cdf='uniform')
    assert pv > 0.05, "`possibly_uniform_distribution` is not a uniform distribution"
    assert (test_range[0] <= possibly_uniform_distribution).all() and \
            (test_range[0] < test_range[1]).all(), \
                "Value returned from the op is outside of requested range"

def test_uniform_continuous_next_after():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    lo = -100
    hi = 100
    test_range = (hi - lo) * np.random.random_sample(2) + lo  # 2 elems from [-100, 100] range
    test_range.sort()
    test_range = test_range.astype('float32')
    with pipe:
        test_range[1] = np.nextafter(test_range[0], test_range[1])
        pipe.set_outputs(dali.fn.uniform(range=test_range.tolist(), shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    assert (test_range[0] == possibly_uniform_distribution).all(), \
                "Value returned from the op is outside of requested range"

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
    possibly_uniform_distribution_chisquare = np.histogram(possibly_uniform_distribution, bins=bins)[0]
    _, pv = st.chisquare(possibly_uniform_distribution_chisquare)
    assert pv > 0.05, "`possibly_uniform_distribution` is not a uniform distribution"
    for val in possibly_uniform_distribution:
        assert val in test_set, \
            "Value returned from the op is outside of requested discrete set"
