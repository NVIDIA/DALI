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


def check_coin_flip(device='cpu', batch_size=32, shape=[1e5], p=None):
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:
        pipe.set_outputs(dali.fn.random.coin_flip(device=device, probability=p, shape=shape))
    pipe.build()
    outputs = pipe.run()
    data_out = outputs[0].as_cpu() \
        if isinstance(outputs[0], TensorListGPU) else outputs[0]
    p = p if p is not None else 0.5
    for i in range(batch_size):
        data = np.array(data_out[i])
        assert np.logical_or(data == 0, data == 1).all()
        total = len(data)
        positive = np.count_nonzero(data)
        np.testing.assert_allclose(p, positive/total, atol=0.005)  # +/- 0.5%

def test_coin_flip():
    batch_size = 8
    shape = [100000]
    for device in ['cpu', 'gpu']:
        for probability in [None, 0.7, 0.5, 0.0, 1.0]:
            yield check_coin_flip, device, batch_size, shape, probability
