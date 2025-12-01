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


from utils import iterator_function_def

from nvidia.dali.plugin.jax import DALIGenericIterator, data_iterator
from test_iterator import run_and_assert_sequential_iterator
from nose2.tools import params

import inspect

# Common parameters for all tests in this file
batch_size = 3


def test_dali_iterator_decorator_functional():
    # given
    iter = data_iterator(iterator_function_def, output_map=["data"], reader_name="reader")(
        batch_size=batch_size, device_id=0, num_threads=4
    )

    # then
    run_and_assert_sequential_iterator(iter)


def test_dali_iterator_decorator_declarative():
    # given
    @data_iterator(output_map=["data"], reader_name="reader")
    def iterator_function():
        return iterator_function_def()

    iter = iterator_function(num_threads=4, device_id=0, batch_size=batch_size)

    # then
    run_and_assert_sequential_iterator(iter)


@params((False,), (True,))
def test_dali_iterator_decorator_declarative_with_default_args(exec_dynamic):
    # given
    @data_iterator(output_map=["data"], reader_name="reader")
    def iterator_function():
        return iterator_function_def()

    iter = iterator_function(batch_size=batch_size, exec_dynamic=exec_dynamic)

    # then
    run_and_assert_sequential_iterator(iter)


@params((False,), (True,))
def test_dali_iterator_decorator_declarative_pipeline_fn_with_argument(exec_dynamic):
    # given
    @data_iterator(output_map=["data"], reader_name="reader")
    def iterator_function(num_shards):
        return iterator_function_def(num_shards=num_shards)

    iter = iterator_function(
        num_shards=2, num_threads=4, device_id=0, batch_size=batch_size, exec_dynamic=exec_dynamic
    )

    # then
    run_and_assert_sequential_iterator(iter)

    # We want to assert that the argument was actually passed. It should affect the
    # number of samples in the iterator.
    # Dataset has 47 samples, with batch_size=3 and num_shards=2, we should get 24 samples.
    # That is because the last batch is extended with the first sample to match the batch_size.
    assert iter.size == 24


# This test checks if the arguments for the iterator decorator match the arguments for
# the iterator __init__ method. Goal is to ensure that the decorator is not missing any
# arguments that might have been added to the iterator __init__
def test_iterator_decorator_api_match_iterator_init():
    # given the list of arguments for the iterator __init__ method
    iterator_init_args = inspect.getfullargspec(DALIGenericIterator.__init__).args
    iterator_init_args.remove("self")
    iterator_init_args.remove("pipelines")

    # given the list of arguments for the iterator decorator
    iterator_decorator_args = inspect.getfullargspec(data_iterator).args
    iterator_decorator_args.remove("pipeline_fn")
    iterator_decorator_args.remove("devices")

    # then
    assert iterator_decorator_args == iterator_init_args, (
        f"Arguments for the iterator decorator and the iterator __init__ method do not match:"
        f"\n------\n{iterator_decorator_args}\n-- vs --\n{iterator_init_args}\n------"
    )

    # Get docs for the decorator "Parameters" section
    # Skip the first argument, which differs (pipelines vs. pipeline_fn)
    # Skip everything after `sharding` argument as it differs between the two
    iterator_decorator_docs = inspect.getdoc(data_iterator)
    iterator_decorator_docs = iterator_decorator_docs.split("output_map")[1]
    iterator_decorator_docs = iterator_decorator_docs.split("sharding")[0]

    # Get docs for the iterator __init__ method "Parameters" section
    iterator_init_docs = inspect.getdoc(DALIGenericIterator)
    iterator_init_docs = iterator_init_docs.split("output_map")[1]
    iterator_init_docs = iterator_init_docs.split("sharding")[0]

    assert iterator_decorator_docs == iterator_init_docs, (
        "Documentation for the iterator decorator and the iterator __init__ method does not match:"
        f"\n------\n{iterator_decorator_docs}\n-- vs --\n{iterator_init_docs}\n------"
    )
