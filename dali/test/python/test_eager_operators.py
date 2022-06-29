# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.eager as eager
import nvidia.dali.tensors as tensors
from nose_utils import assert_raises, raises


@raises(RuntimeError, glob="Argument '*' is not supported by eager operator 'crop'.")
def _test_disqualified_argument(key):
    tl = tensors.TensorListCPU(np.zeros((8, 256, 256, 3)))
    eager.crop(tl, crop=[64, 64], **{key: 0})


def test_disqualified_arguments():
    for arg in ['bytes_per_sample_hint', 'preserve', 'seed']:
        yield _test_disqualified_argument, arg


@raises(TypeError, glob="unsupported operand type*")
def test_arithm_op_context_manager_disabled():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))

    tl_1 + tl_2


def test_arithm_op_context_manager_enabled():
    eager.arithmetic(True)
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))

    assert np.array_equal((tl_1 + tl_2).as_array(), np.full(shape=(8, 16, 16), fill_value=2))
    eager.arithmetic(False)


def test_arithm_op_context_manager_nested():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    expected_sum = np.full(shape=(8, 16, 16), fill_value=2)

    with eager.arithmetic():
        assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

        with eager.arithmetic(False):
            with assert_raises(TypeError, glob="unsupported operand type*"):
                tl_1 + tl_2

        assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)


def test_arithm_op_context_manager_deep_nested():
    tl_1 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    tl_2 = tensors.TensorListCPU(np.ones((8, 16, 16)))
    expected_sum = np.full(shape=(8, 16, 16), fill_value=2)

    eager.arithmetic(True)

    assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

    with eager.arithmetic(False):
        with assert_raises(TypeError, glob="unsupported operand type*"):
            tl_1 + tl_2

        with eager.arithmetic(True):
            np.array_equal((tl_1 + tl_2).as_array(), expected_sum)

            with eager.arithmetic(False):
                with assert_raises(TypeError, glob="unsupported operand type*"):
                    tl_1 + tl_2

        with assert_raises(TypeError, glob="unsupported operand type*"):
            tl_1 + tl_2

    assert np.array_equal((tl_1 + tl_2).as_array(), expected_sum)
    eager.arithmetic(False)
