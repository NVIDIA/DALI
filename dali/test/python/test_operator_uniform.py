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
import numpy as np
import scipy.stats as st
import math



def test_uniform_default():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    with pipe:
        pipe.set_outputs(dali.fn.uniform(shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    _, pv = st.kstest(rvs=possibly_uniform_distribution, cdf='uniform')
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
    assert (-1 <= possibly_uniform_distribution).all() and \
            (possibly_uniform_distribution < 1).all(), \
                "Value returned from the op is outside of requested range"


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
    _, pv = st.kstest(rvs=possibly_uniform_distribution, cdf='uniform')
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
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
    _, pv = st.kstest(rvs=possibly_uniform_distribution, cdf='uniform')
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
    assert (test_range[0] <= possibly_uniform_distribution).all() and \
            (test_range[0] < test_range[1]).all(), \
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
    _, pv = st.kstest(rvs=possibly_uniform_distribution, cdf='uniform')
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
    for val in possibly_uniform_distribution:
        assert val in test_set, \
            "Value returned from the op is outside of requested discrete set"

