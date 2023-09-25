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

from utils import iterator_function_def

from nvidia.dali.plugin.jax import DALIGenericIterator, data_iterator
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from nose_utils import raises

import inspect
import itertools

# Common parameters for all tests in this file
batch_size = 3


def run_and_assert_sequential_iterator(iter, num_iters=4):
    """Run the iterator and assert that the output is as expected"""
    # when
    for batch_id, data in itertools.islice(enumerate(iter), num_iters):
        jax_array = data['data']

        # then
        assert jax_array.device() == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i],
                jax.numpy.full(
                    (1),
                    batch_id * batch_size + i,
                    np.int32))

    assert batch_id == num_iters - 1


def test_dali_sequential_iterator():
    # given
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iter = DALIGenericIterator([pipe], ['data'], reader_name='reader')

    # then
    run_and_assert_sequential_iterator(iter)


@raises(AssertionError, glob="JAX iterator does not support partial last batch policy.")
def test_iterator_last_batch_policy_partial_exception():
    pipe = pipe = pipeline_def(iterator_function_def)(
        batch_size=batch_size, num_threads=4, device_id=0)
    DALIGenericIterator(
        [pipe], ['data'], reader_name='reader', last_batch_policy=LastBatchPolicy.PARTIAL)


def test_dali_iterator_decorator_all_pipeline_args_in_decorator():
    # given
    iter = data_iterator(
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
    iter = data_iterator(
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
    iter = data_iterator(
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
    @data_iterator(
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
    @data_iterator(
        output_map=['data'],
        reader_name='reader',
        num_threads=4,
        device_id=0,
        batch_size=batch_size)
    def iterator_function(num_shards):
        return iterator_function_def(num_shards=num_shards)

    iter = iterator_function(num_shards=2)

    # then
    run_and_assert_sequential_iterator(iter)

    # We want to assert that the argument was actually passed. It should affect the
    # number of samples in the iterator.
    # Dataset has 47 samples, with batch_size=3 and num_shards=2, we should get 24 samples.
    # That is because the last batch is extended with the first sample to match the batch_size.
    assert iter.size == 24


@raises(ValueError,  glob="Duplicate argument batch_size in decorator and a call")
def test_iterator_decorator_pipeline_arg_duplicate():
    data_iterator(
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
    iterator_init_args = inspect.getfullargspec(DALIGenericIterator.__init__).args
    iterator_init_args.remove("self")
    iterator_init_args.remove("pipelines")

    # given the list of arguments for the iterator decorator
    iterator_decorator_args = inspect.getfullargspec(data_iterator).args
    iterator_decorator_args.remove("pipeline_fn")

    # then
    assert iterator_decorator_args == iterator_init_args, \
        "Arguments for the iterator decorator and the iterator __init__ method do not match"

    # TODO: Do we want to test documentation as well?
