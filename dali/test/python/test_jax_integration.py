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

from nvidia.dali import pipeline_def
from nvidia.dali.backend import TensorGPU
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import nvidia.dali.plugin.jax as dax

from nose2.tools import cartesian_params


def get_dali_tensor_gpu(value, shape, dtype) -> TensorGPU:
    """Helper function to create DALI TensorGPU.

    Args:
        value : Value to fill the tensor with.
        shape : Shape for the tensor.
        dtype : Data type for the tensor.

    Returns:
        TensorGPU: DALI TensorGPU with provided shape and dtype filled
        with provided value.
    """
    @pipeline_def(num_threads=1, batch_size=1)
    def dali_pipeline():
        values = fn.constant(idata=value, shape=shape, dtype=dtype, device='gpu')

        return values

    pipe = dali_pipeline(device_id=0)
    pipe.build()
    dali_output = pipe.run()

    return dali_output[0][0]


@cartesian_params(
    (types.FLOAT, types.INT32),           # dtypes to test
    ([], [1], [10], [2, 4], [1, 2, 3]),   # shapes to test
    (1, -99))                             # values to test
def test_dali_tensor_gpu_to_jax_array(dtype, shape, value):
    # given
    dali_tensor_gpu = get_dali_tensor_gpu(
        value=value, shape=shape, dtype=dtype)

    # when
    jax_array = dax._to_jax_array(dali_tensor_gpu)

    # then
    assert jax.numpy.array_equal(
        jax_array,
        jax.numpy.full(shape, value))

    # Make sure JAX array is backed by the GPU
    assert jax_array.device() == jax.devices()[0]


def test_dali_sequential_tensors_to_jax_array():
    batch_size = 4
    shape = (1, 5)

    def numpy_sequential_tensors(sample_info):
        return np.full(shape, sample_info.idx_in_epoch, dtype=np.int32)

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def callable_pipeline():
        data = fn.external_source(
            source=numpy_sequential_tensors,
            num_outputs=1,
            batch=False,
            dtype=types.INT32)
        data = data[0].gpu()
        return data

    pipe = callable_pipeline()
    pipe.build()

    for batch_id in range(100):
        # given
        dali_tensor_gpu = pipe.run()[0].as_tensor()

        # when
        jax_array = dax._to_jax_array(dali_tensor_gpu)

        # then
        assert jax_array.device() == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    shape[1:],  # TODO(awolant): Explain/fix shape consistency
                    batch_id * batch_size + i,
                    np.int32))


def test_dali_sequential_iterator_to_jax_array():
    batch_size = 4
    shape = (1, 5)

    def numpy_sequential_tensors(sample_info):
        return np.full(shape, sample_info.idx_in_epoch, dtype=np.int32)

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def callable_pipeline():
        data = fn.external_source(
            source=numpy_sequential_tensors,
            num_outputs=1,
            batch=False,
            dtype=types.INT32)
        data = data[0].gpu()
        return data

    pipe = callable_pipeline()
    iter = dax.DALIGenericIterator([pipe], ['data'], size=batch_size*100)

    for batch_id, data in enumerate(iter):
        # given
        jax_array = data[0]['data']

        # then
        assert jax_array.device() == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    shape[1:],  # TODO(awolant): Explain shape consistency
                    batch_id * batch_size + i,
                    np.int32))
