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
from nvidia.dali.plugin.jax.clu import DALIGenericPeekableIterator as DALIIterator
from test_integration import sequential_pipeline
from clu.data.dataset_iterator import ArraySpec

from nose_utils import raises
import time

from utils import pipeline_with_variable_shape_output

# Common parameters for all tests in this file
batch_size = 3
shape = (1, 5)
batch_shape = (batch_size, *shape[1:])


def test_jax_peekable_iterator_peek():
    # given
    pipe = sequential_pipeline(batch_size, shape)

    # when
    iterator = DALIIterator([pipe], ['data'], size=batch_size*100)

    # then
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}

    for i in range(5):
        peeked_output = iterator.peek()
        output = iterator.next()

        assert jnp.array_equal(
            output['data'], peeked_output['data'])


def test_jax_peekable_iterator_peek_async_result_before_next():
    # given
    pipe = sequential_pipeline(batch_size, shape)

    # when
    iterator = DALIIterator([pipe], ['data'], size=batch_size*100)

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
    pipe = sequential_pipeline(batch_size, shape)

    # when
    iterator = DALIIterator([pipe], ['data'], size=batch_size*100)

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

    iterator = DALIIterator([pipe], ['data'], size=batch_size*100)
    iterator.next()

    # when
    iterator.next()
