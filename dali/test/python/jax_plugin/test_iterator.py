# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import jax
import jax.numpy
import jax.dlpack

from utils import sequential_pipeline

import nvidia.dali.plugin.jax as dax


def test_dali_sequential_iterator_to_jax_array():
    batch_size = 4
    shape = (1, 5)

    pipe = sequential_pipeline(batch_size, shape)
    iter = dax.DALIGenericIterator([pipe], ['data'], size=batch_size*100)

    for batch_id, data in enumerate(iter):
        # given
        jax_array = data['data']

        # then
        assert jax_array.device() == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    shape[1:],  # TODO(awolant): Explain shape consistency
                    batch_id * batch_size + i,
                    np.int32))

    assert batch_id == 99
