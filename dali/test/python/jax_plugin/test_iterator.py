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

from utils import sequential_pipeline, sequential_pipeline_def

import nvidia.dali.plugin.jax as dax

from nose_utils import raises

import inspect

# Common parameters for all tests in this file
batch_size = 3
shape = (1, 5)
iterator_size = batch_size*10


def test_dali_sequential_iterator_to_jax_array():
    # given
    pipe = sequential_pipeline(batch_size, shape)
    iter = dax.DALIGenericIterator([pipe], ['data'], size=iterator_size)

    # when
    for batch_id, data in enumerate(iter):
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

    assert batch_id == 9


def test_dali_iterator_decorator():
    # given
    iter = dax.iterator.data_iterator(
        sequential_pipeline_def,
        output_map=['data'],
        batch_size=batch_size,
        device_id=0,
        size=iterator_size)(num_threads=4)

    # when
    for batch_id, data in enumerate(iter):
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

    assert batch_id == 9


@raises(ValueError,  glob="Duplicate argument batch_size in decorator and a call")
def test_iterator_decorator_pipeline_arg_duplicate():
    dax.iterator.data_iterator(
        sequential_pipeline_def,
        output_map=['data'],
        batch_size=4,
        device_id=0,
        size=iterator_size)(num_threads=4, batch_size=1000, pipeline_fn=sequential_pipeline_def)


def test_iterator_decorator_kwargs_match_iterator_init():
    # get the list of arguments for the iterator __init__ method
    iterator_init_args = inspect.getfullargspec(dax.iterator.DALIGenericIterator.__init__).args
    iterator_init_args.remove("self")
    iterator_init_args.remove("pipelines")

    # get the list of arguments for the iterator decorator
    iterator_decorator_args = inspect.getfullargspec(dax.iterator.data_iterator).args
    iterator_decorator_args.remove("pipeline_fn")

    assert iterator_decorator_args == iterator_init_args, \
        "Arguments for the iterator decorator and the iterator __init__ method do not match"
