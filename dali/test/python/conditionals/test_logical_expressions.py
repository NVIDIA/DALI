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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from test_utils import compare_pipelines, check_batch
from nose_utils import assert_raises
from nose2.tools import params

import numpy as np


def test_not():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(**kwargs)
    def regular_pipe():
        boolean_input = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=42)
        return boolean_input == 0

    @pipeline_def(enable_conditionals=True, **kwargs)
    def not_pipe():
        boolean_input = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=42)
        return not boolean_input

    pipes = [regular_pipe(), not_pipe()]
    compare_pipelines(*pipes, bs, iters)


def test_and():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(**kwargs)
    def regular_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        const_F = types.Constant(np.array(False), device="cpu")
        const_T = types.Constant(np.array(True), device="cpu")
        return (
            boolean_input_0 & boolean_input_1,
            boolean_input_0 & const_F,
            const_T & boolean_input_1,
        )

    @pipeline_def(enable_conditionals=True, **kwargs)
    def and_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        const_F = types.Constant(np.array(False), device="cpu")
        const_T = types.Constant(np.array(True), device="cpu")
        return (
            boolean_input_0 and boolean_input_1,
            boolean_input_0 and const_F,
            const_T and boolean_input_1,
        )

    pipes = [regular_pipe(), and_pipe()]
    compare_pipelines(*pipes, bs, iters)


def test_or():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(**kwargs)
    def regular_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        const_F = types.Constant(np.array(False), device="cpu")
        const_T = types.Constant(np.array(True), device="cpu")
        return (
            boolean_input_0 | boolean_input_1,
            boolean_input_0 | const_F,
            const_T | boolean_input_1,
        )

    @pipeline_def(enable_conditionals=True, **kwargs)
    def or_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        const_F = types.Constant(np.array(False), device="cpu")
        const_T = types.Constant(np.array(True), device="cpu")
        return (
            boolean_input_0 or boolean_input_1,
            boolean_input_0 or const_F,
            const_T or boolean_input_1,
        )

    pipes = [regular_pipe(), or_pipe()]
    compare_pipelines(*pipes, bs, iters)


def test_complex_expression():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(**kwargs)
    def regular_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        boolean_input_2 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=12)
        boolean_input_3 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=15)
        return (boolean_input_0 | (boolean_input_1 & boolean_input_2)) | (boolean_input_3 == 0)

    @pipeline_def(enable_conditionals=True, **kwargs)
    def expr_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        boolean_input_2 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=12)
        boolean_input_3 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=15)
        return boolean_input_0 or boolean_input_1 and boolean_input_2 or not boolean_input_3

    pipes = [regular_pipe(), expr_pipe()]
    compare_pipelines(*pipes, bs, iters)


def test_lazy_eval():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(enable_conditionals=True, **kwargs)
    def if_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        if boolean_input_0:
            val = boolean_input_1 == False  # noqa: E712
        else:
            val = boolean_input_0
        return val

    @pipeline_def(enable_conditionals=True, **kwargs)
    def expr_pipe():
        boolean_input_0 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=6)
        boolean_input_1 = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, seed=9)
        val = boolean_input_0 and boolean_input_1 == False  # noqa: E712
        return val

    pipes = [if_pipe(), expr_pipe()]
    compare_pipelines(*pipes, bs, iters)


def test_lazy_eval_with_oob():
    bs = 10
    iters = 5
    kwargs = {"batch_size": bs, "num_threads": 4, "device_id": 0, "seed": 42}

    @pipeline_def(enable_conditionals=True, **kwargs)
    def base_pipe():
        return types.Constant(np.bool_(True))

    @pipeline_def(enable_conditionals=True, **kwargs)
    def expr_pipe():
        boolean_tensor_input = types.Constant(np.bool_([True, True, False]), device="cpu")
        index_input_1 = types.Constant(np.int32(1), device="cpu")
        index_input_42 = types.Constant(np.int32(42), device="cpu")
        # do an oob access in the right subexpression that won't be evaluated.
        val = boolean_tensor_input[index_input_1] or boolean_tensor_input[index_input_42]
        return val

    pipes = [base_pipe(), expr_pipe()]
    compare_pipelines(*pipes, bs, iters)


def logical_true_false_random():
    """Return an External Source that returns batch of [True, False, <random booleans>]
    so we always have at least one sample that is True or False.
    Otherwise we may end up with fully short-cutting part of the expression we want to test.
    """
    rng = np.random.default_rng(seed=101)

    def get_true_false_random(sample_info):
        if sample_info.idx_in_batch == 0:
            return np.array(True)
        elif sample_info.idx_in_batch == 1:
            return np.array(False)
        else:
            return rng.choice([np.array(True), np.array(False)])

    return fn.external_source(source=get_true_false_random, batch=False)


logical_expressions = [
    lambda x: not x,
    lambda x: x and logical_true_false_random(),
    lambda x: logical_true_false_random() and x,
    lambda x: x or logical_true_false_random(),
    lambda x: logical_true_false_random() or x,
]


@params(*logical_expressions)
def test_error_input(expression):
    kwargs = {
        "enable_conditionals": True,
        "batch_size": 10,
        "num_threads": 4,
        "device_id": 0,
    }

    @pipeline_def(**kwargs)
    def gpu_input():
        input = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        return expression(input.gpu())

    # We can make a valid graph with `not` op directly, the rest (`and`, `or`) is basically lowered
    # to `if` statements and thus checked by graph via argument input placement validation.
    with assert_raises(
        RuntimeError,
        regex=(
            "Logical expression `.*` is restricted to scalar \\(0-d tensors\\)"
            " inputs of `bool` type, that are placed on CPU."
            " Got a GPU input .*in logical expression.*|"
            "Named argument inputs to operators must be CPU data nodes."
            " However, a GPU data node was provided"
        ),
    ):
        pipe = gpu_input()
        pipe.run()

    @pipeline_def(**kwargs)
    def non_scalar_input():
        pred = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        stacked = fn.stack(pred, pred)
        return expression(stacked)

    with assert_raises(
        RuntimeError,
        glob=(
            "Logical expression `*` is restricted to scalar (0-d tensors)"
            " inputs*, that are placed on CPU. Got a 1-d input"
            " *in logical expression."
        ),
    ):
        pipe = non_scalar_input()
        pipe.run()


boolean_restricted_logical_expressions = [
    lambda x: x and logical_true_false_random(),
    lambda x: logical_true_false_random() and x,
    lambda x: x or logical_true_false_random(),
    lambda x: logical_true_false_random() or x,
]


@params(*boolean_restricted_logical_expressions)
def test_non_boolean_input_error(expression):
    kwargs = {
        "enable_conditionals": True,
        "batch_size": 10,
        "num_threads": 4,
        "device_id": 0,
    }

    @pipeline_def(**kwargs)
    def non_bool_input():
        input = fn.random.coin_flip(dtype=types.DALIDataType.INT32)
        return expression(input)

    with assert_raises(
        RuntimeError,
        glob=(
            "Logical expression `*` is restricted to scalar (0-d tensors)"
            " inputs of `bool` type, that are placed on CPU. Got an input"
            " of type `int32` *in logical expression."
        ),
    ):
        pipe = non_bool_input()
        pipe.run()


boolable_types = [
    bool,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


@params(*boolable_types)
def test_not_any_type(input_type):
    batch_size = 10
    kwargs = {
        "enable_conditionals": True,
        "batch_size": batch_size,
        "num_threads": 4,
        "device_id": 0,
    }

    def get_truthy_falsy(sample_info):
        if sample_info.idx_in_batch < batch_size / 2:
            return np.array(42, dtype=input_type)
        else:
            return np.array(0, dtype=input_type)

    @pipeline_def(**kwargs)
    def non_bool_input():
        input = fn.external_source(source=get_truthy_falsy, batch=False)
        return not input

    pipe = non_bool_input()
    (batch,) = pipe.run()

    target = [False if i < batch_size / 2 else True for i in range(batch_size)]
    check_batch(batch, target)
