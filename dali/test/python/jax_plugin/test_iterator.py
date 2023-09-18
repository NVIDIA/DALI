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

from utils import iterator_function_def, sequential_dataset

import nvidia.dali.plugin.jax as dax
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from nose_utils import raises

import inspect
import itertools

# Common parameters for all tests in this file
batch_size = 3


def run_and_assert_sequential_iterator(iter):
    """Run the iterator and assert that the output is as expected"""
    # when
    for batch_id, data in itertools.islice(enumerate(iter), 10):
        jax_array = data['data']

        # then
        assert jax_array.device() == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    sequential_dataset['sample_shape'],
                    batch_id * batch_size + i,
                    np.int32))

    assert batch_id == 9


def test_dali_sequential_iterator():
    # given
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iter = dax.DALIGenericIterator([pipe], ['data'], reader_name='reader')

    # then
    run_and_assert_sequential_iterator(iter)


@raises(AssertionError, glob="JAX iterator does not support partial last batch policy.")
def test_iterator_last_batch_policy_partial_exception():
    pipe = pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    dax.DALIGenericIterator(
        [pipe], ['data'], reader_name='reader', last_batch_policy=LastBatchPolicy.PARTIAL)


def test_dali_iterator_decorator_all_pipeline_args_in_decorator():
    # given
    iter = dax.iterator.data_iterator(
        iterator_function_def,
        output_map=['data'],
        batch_size=batch_size,
        device_id=0,
        num_threads=4,
        reader_name='reader')()

    # then
    run_and_assert_sequential_iterator(iter)


def test_dali_iterator_decorator_all_pipeline_args_in_call():
    # given
    iter = dax.iterator.data_iterator(
        iterator_function_def,
        output_map=['data'],
        reader_name='reader')(
            batch_size=batch_size,
            device_id=0,
            num_threads=4)

    # then
    run_and_assert_sequential_iterator(iter)


def test_dali_iterator_decorator_pipeline_args_split_in_decorator_and_call():
    # given
    iter = dax.iterator.data_iterator(
        iterator_function_def,
        output_map=['data'],
        reader_name='reader',
        num_threads=4,
        device_id=0)(
            batch_size=batch_size)

    # then
    run_and_assert_sequential_iterator(iter)


def test_dali_iterator_decorator_declarative():
    # given
    @dax.iterator.data_iterator(
        output_map=['data'],
        reader_name='reader',
        num_threads=4,
        device_id=0,
        batch_size=batch_size)
    def iterator_function():
        return iterator_function_def()

    iter = iterator_function()

    # then
    run_and_assert_sequential_iterator(iter)


def test_dali_iterator_decorator_declarative_pipeline_fn_with_argument():
    # given
    @dax.iterator.data_iterator(
        output_map=['data'],
        reader_name='reader',
        num_threads=4,
        device_id=0,
        batch_size=batch_size)
    def iterator_function(dataset_file_name):
        return iterator_function_def(dataset_file_name=dataset_file_name)

    iter = iterator_function(dataset_file_name='sequential.tfrecord')

    # then
    run_and_assert_sequential_iterator(iter)


@raises(ValueError,  glob="Duplicate argument batch_size in decorator and a call")
def test_iterator_decorator_pipeline_arg_duplicate():
    dax.iterator.data_iterator(
        iterator_function_def,
        output_map=['data'],
        batch_size=4,
        device_id=0,
        reader_name='reader')(
            num_threads=4, batch_size=1000)


# This test checks if the arguments for the iterator decorator match the arguments for
# the iterator __init__ method. Goal is to ensure that the decorator is not missing any
# arguments that might have been added to the iterator __init__
def test_iterator_decorator_kwargs_match_iterator_init():
    # given the list of arguments for the iterator __init__ method
    iterator_init_args = inspect.getfullargspec(dax.iterator.DALIGenericIterator.__init__).args
    iterator_init_args.remove("self")
    iterator_init_args.remove("pipelines")

    # given the list of arguments for the iterator decorator
    iterator_decorator_args = inspect.getfullargspec(dax.iterator.data_iterator).args
    iterator_decorator_args.remove("pipeline_fn")

    # then
    assert iterator_decorator_args == iterator_init_args, \
        "Arguments for the iterator decorator and the iterator __init__ method do not match"
