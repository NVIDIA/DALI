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


def test_uniform_continuous():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    test_range = (-100 - 100) * np.random.random_sample(2) + 100  # 2 elems from [-100, 100] range
    with pipe:
        pipe.set_outputs(dali.fn.uniform(range=test_range.tolist(), shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    _, pv = st.chisquare(possibly_uniform_distribution, axis=None)
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
    for val in possibly_uniform_distribution:
        assert test_range[0] <= val < test_range[1], \
            "Value returned from the op is outside of requested range"


def in_float_set(val, test_set):
    for t in test_set:
        if math.isclose(val, t, rel_tol=1e-2):
            return True
    print(val, test_set)
    return False


def test_uniform_discreet():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    test_set = (-100 - 100) * np.random.random_sample(10) + 100  # 10 elems from [-100, 100] range
    with pipe:
        pipe.set_outputs(dali.fn.uniform(set=test_set.tolist(), shape=[1e6]))
    pipe.build()
    oo = pipe.run()
    possibly_uniform_distribution = oo[0].as_array()[0]
    _, pv = st.chisquare(possibly_uniform_distribution, axis=None)
    print(pv, test_set)
    # if pv >1e-8:
    #     import ipdb; ipdb.set_trace()
    assert pv < 1e-8, "`possibly_uniform_distribution` is not an uniform distribution"
    # import ipdb; ipdb.set_trace()
    for val in possibly_uniform_distribution:
        assert in_float_set(val, test_set), \
            "Value returned from the op is outside of requested discrete set"

if __name__ == '__main__':
    test_uniform_continuous()
