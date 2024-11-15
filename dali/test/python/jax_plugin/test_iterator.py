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

from utils import iterator_function_def

import nvidia.dali.plugin.jax as dax
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nose_utils import raises
from nose2.tools import params

import itertools

# Common parameters for all tests in this file
batch_size = 3


def run_and_assert_sequential_iterator(iter, num_iters=4):
    """Run the iterator and assert that the output is as expected"""
    # when
    for batch_id, data in itertools.islice(enumerate(iter), num_iters):
        jax_array = data["data"]

        # then
        assert dax.integration._jax_device(jax_array) == jax.devices()[0]

        for i in range(batch_size):
            assert jax.numpy.array_equal(
                jax_array[i], jax.numpy.full((1), batch_id * batch_size + i, np.int32)
            )

    assert batch_id == num_iters - 1


@params((False,), (True,))
def test_dali_sequential_iterator(exec_dynamic):
    # given
    pipe = pipeline_def(iterator_function_def)(
        batch_size=batch_size, num_threads=4, device_id=0, exec_dynamic=exec_dynamic
    )
    iter = DALIGenericIterator([pipe], ["data"], reader_name="reader")

    # then
    run_and_assert_sequential_iterator(iter)


@raises(AssertionError, glob="JAX iterator does not support partial last batch policy.")
def test_iterator_last_batch_policy_partial_exception():
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    DALIGenericIterator(
        [pipe], ["data"], reader_name="reader", last_batch_policy=LastBatchPolicy.PARTIAL
    )
