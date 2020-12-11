# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import scipy.stats as st
import random

test_types = [types.INT8, types.INT16, types.INT32, types.INT16, types.FLOAT, types.FLOAT64, types.FLOAT16]

def check_normal_distribution(device, dtype, shape=None, use_shape_like_input=False, mean=0.0, stddev=1.0,
                              niter=3, batch_size=3, device_id=0, num_threads=3):
    pipe = Pipeline(batch_size=batch_size, device_id=device_id, num_threads=num_threads, seed=123456)
    with pipe:
        if shape is not None:
            if use_shape_like_input:
                out = fn.random.normal(np.ones(shape), device=device, mean=mean, stddev=stddev, dtype=dtype)
            else:
                out = fn.random.normal(device=device, shape=shape, mean=mean, stddev=stddev, dtype=dtype)
        else:
            out = fn.random.normal(device=device, mean=mean, stddev=stddev, dtype=dtype)
        pipe.set_outputs(out)
    pipe.build()
    for i in range(niter):
        outputs = pipe.run()
        out = outputs[0] if device == 'cpu' else outputs[0].as_cpu()
        for s in range(batch_size):
            sample = out.at(s)
            if shape is not None:
                assert sample.shape == shape, f"{sample.shape} != {shape}"
            else:
                assert sample.shape == (1,), f"{sample.shape} != (1, )"

            data = sample.flatten()

            m = np.mean(data)
            s = np.std(data)
            l = len(data)
            # Checking sanity of the data
            if l >= 100 and dtype in [types.FLOAT, types.FLOAT64, types.FLOAT16]:
                # Empirical rule: 
                # ~68% of the observations within one standard deviation
                # ~95% of the observations within one standard deviation
                # ~99.7% of the observations within one standard deviation
                within_1stddevs = np.where((data > (mean - 1 * stddev)) & (data < (mean + 1 * stddev)))
                p1 = len(within_1stddevs[0]) / l
                within_2stddevs = np.where((data > (mean - 2 * stddev)) & (data < (mean + 2 * stddev)))
                p2 = len(within_2stddevs[0]) / l
                within_3stddevs = np.where((data > (mean - 3 * stddev)) & (data < (mean + 3 * stddev)))
                p3 = len(within_3stddevs[0]) / l
                assert p3 > 0.9,  f"{p3}"   # leave some room
                assert p2 > 0.8,  f"{p2}"   # leave some room
                assert p1 > 0.5,  f"{p1}"   # leave some room

            # It's not 100% mathematically correct, but makes do in case of this test
            _, pvalues_anderson, _ = st.anderson(data, dist='norm')
            assert pvalues_anderson[2] > 0.5

def test_normal_distribution_single_value():
    for device in ("cpu", "gpu"):
        for dtype in test_types:
            for shape in [(100,), (10, 20, 30), (1, 2, 3, 4, 5, 6)]:
                use_shape_like_in = random.choice([True, False])
                for mean, stddev in [(0.0, 1.0), (111.0, 57.0)]:
                    yield check_normal_distribution, device, dtype, shape, use_shape_like_in, mean, stddev
