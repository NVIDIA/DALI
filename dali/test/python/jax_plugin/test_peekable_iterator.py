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


import jax.numpy as jnp
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.jax.clu import DALIGenericPeekableIterator as DALIPeekableIterator
from nvidia.dali.plugin.jax.clu import peekable_data_iterator
from nvidia.dali.plugin.jax.iterator import DALIGenericIterator
from utils import iterator_function_def
from clu.data.dataset_iterator import ArraySpec

from test_iterator import run_and_assert_sequential_iterator

from nose_utils import raises
import time

import inspect

from utils import pipeline_with_variable_shape_output

# Common parameters for all tests in this file
batch_size = 3
batch_shape = (batch_size, 1)


def test_jax_peekable_iterator_peek():
    # given
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)

    # when
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')

    # then
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}

    for i in range(5):
        peeked_output = iterator.peek()
        output = iterator.next()

        assert jnp.array_equal(
            output['data'], peeked_output['data'])


def test_jax_peekable_iterator_peek_async_result_before_next():
    # given
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)

    # when
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')

    # then
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}

    for i in range(5):
        peeked_output = iterator.peek_async()
        peeked_output = peeked_output.result()
        output = iterator.next()

        assert jnp.array_equal(
            output['data'], peeked_output['data']), \
            f"output: {output['data']}, peeked_output: {peeked_output['data']}"


def test_jax_peekable_iterator_peek_async_result_after_next():
    '''This test is not deterministic, but it should pass most of the time.'''
    # given
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)

    # when
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')

    # then
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}

    for i in range(5):
        peeked_output = iterator.peek_async()
        time.sleep(0.1)  # wait before calling next to give time for peek to start
        output = iterator.next()
        peeked_output = peeked_output.result()

        assert jnp.array_equal(
            output['data'], peeked_output['data']), \
            f"output: {output['data']}, peeked_output: {peeked_output['data']}"


@raises(ValueError, glob="The shape or type of the output changed between iterations.")
def test_jax_peekable_iterator_with_variable_shapes_pipeline():
    # given
    batch_size = 1
    pipe = pipeline_with_variable_shape_output(batch_size)

    iterator = DALIPeekableIterator([pipe], ['data'], size=batch_size*100)
    iterator.next()

    # when
    iterator.next()


# Makes sure that API of DALIGenericIterator and DALIGenericPeekableIterator did not diverge
def test_iterators_init_method_args_compatibility():
    # given
    iterator_init_args = inspect.getfullargspec(DALIGenericIterator.__init__).args
    peekalbe_iterator_init_args = inspect.getfullargspec(DALIPeekableIterator.__init__).args

    # then
    assert iterator_init_args == peekalbe_iterator_init_args


def test_dali_iterator_decorator_all_pipeline_args_in_decorator():
    # given
    iter = peekable_data_iterator(
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
    iter = peekable_data_iterator(
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
    iter = peekable_data_iterator(
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
    @peekable_data_iterator(
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
    @peekable_data_iterator(
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
    peekable_data_iterator(
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
    iterator_init_args = inspect.getfullargspec(DALIPeekableIterator.__init__).args
    iterator_init_args.remove("self")
    iterator_init_args.remove("pipelines")

    # given the list of arguments for the iterator decorator
    iterator_decorator_args = inspect.getfullargspec(peekable_data_iterator).args
    iterator_decorator_args.remove("pipeline_fn")

    # then
    assert iterator_decorator_args == iterator_init_args, \
        "Arguments for the iterator decorator and the iterator __init__ method do not match"
