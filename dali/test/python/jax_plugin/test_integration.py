# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.plugin.jax as dax

from nose2.tools import cartesian_params
from utils import get_dali_tensor_gpu, sequential_pipeline


@cartesian_params(
    (np.float32, np.int32),  # dtypes to test
    ([], [1], [10], [2, 4], [1, 2, 3]),  # shapes to test
    (1, -99),
)  # values to test
def test_dali_tensor_gpu_to_jax_array(dtype, shape, value):
    # given
    dali_tensor_gpu = get_dali_tensor_gpu(value=value, shape=shape, dtype=dtype)

    # when
    jax_array = dax.integration._to_jax_array(dali_tensor_gpu, False)

    # then
    assert jax.numpy.array_equal(jax_array, jax.numpy.full(shape, value, dtype))

    # Make sure JAX array is backed by the GPU
    assert dax.integration._jax_device(jax_array) == jax.devices()[0]


def test_dali_sequential_tensors_to_jax_array():
    batch_size = 4
    shape = (1, 5)

    pipe = sequential_pipeline(batch_size, shape)

    for batch_id in range(100):
        # given
        dali_tensor_gpu = pipe.run()[0].as_tensor()

        # when
        jax_array = dax.integration._to_jax_array(dali_tensor_gpu, False)

        # then
        assert dax.integration._jax_device(jax_array) == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    shape[1:],  # TODO(awolant): Explain/fix shape consistency
                    batch_id * batch_size + i,
                    np.int32,
                ),
            )
