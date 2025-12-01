# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def, fn, types
from test_utils import load_test_operator_plugin
from nose_utils import assert_raises
from nose2.tools import params
from nvidia.dali import math


def setUpModule():
    load_test_operator_plugin()


python_errors = [RuntimeError, IndexError, TypeError, ValueError, StopIteration]


def test_python_error_constructor():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(constructor=True)

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error when building pipeline:
Error in CPU operator `nvidia.dali.fn.throw_exception`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(constructor=True)

encountered:

Error in constructor
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


@params(*python_errors)
def test_python_error_propagation(error):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type=error.__name__)

    with assert_raises(
        error,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.fn.throw_exception`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type=error.__name__)

encountered:

Test message
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


pybind_mapped_errors = [
    ("std::runtime_error", RuntimeError),
    ("std::domain_error", ValueError),
    ("std::invalid_argument", ValueError),
    ("std::length_error", ValueError),
    ("std::out_of_range", IndexError),
    ("std::range_error", ValueError),
]


@params(*pybind_mapped_errors)
def test_cpp_error_propagation(error_name, error_type):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type=error_name)

    with assert_raises(
        error_type,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.fn.throw_exception`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type=error_name)

encountered:

Test message
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


def test_error_propagation_ellipsis():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type="std::string")

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.fn.throw_exception`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type="std::string")

encountered:

Unknown critical error.
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


def test_arithm_ops():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        a = fn.random.uniform(range=[-1, 1], shape=(2, 3))
        b = fn.random.uniform(range=[-1, 1], shape=(3, 2))
        return a + b

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.math.add`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return a + b

encountered:

Can't broadcast shapes*
2 x 3 (d=1, belonging to sample_idx=0)
3 x 2 (d=1, belonging to sample_idx=0)

*C++ context: *broadcasting*
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


def test_math_ops():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        a = fn.random.uniform(range=[-1, 1], shape=(2, 3))
        b = fn.random.uniform(range=[-1, 1], shape=(3, 2))
        return math.atan2(a, b) + 1

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.math.atan2`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in pipe
    return math.atan2(a, b) + 1

encountered:

Can't broadcast shapes*
2 x 3 (d=1, belonging to sample_idx=0)
3 x 2 (d=1, belonging to sample_idx=0)

*C++ context: *broadcasting*
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.run()


def test_conditional_split():
    @pipeline_def(enable_conditionals=True, batch_size=10, num_threads=4, device_id=0)
    def non_scalar_condition():
        pred = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        stacked = fn.stack(pred, pred)
        if stacked:
            output = types.Constant([1])
        else:
            output = types.Constant([0])
        return output

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.fn._conditional.split`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in non_scalar_condition
    if stacked:

encountered:

*Assert on "dim == 0" failed: Conditions inside `if` statements are restricted to scalar*"""
        ),
    ):
        pipe = non_scalar_condition()
        pipe.run()


def test_conditional_merge():
    @pipeline_def(enable_conditionals=True, batch_size=10, num_threads=4, device_id=0)
    def non_scalar_condition():
        pred = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        if pred:
            output = types.Constant([1])
        else:
            output = types.Constant([1.0])
        return output

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error in CPU operator `nvidia.dali.fn._conditional.merge`,
which was used in the pipeline definition with the following traceback:

  File "*test_operator_exception.py", line *, in non_scalar_condition
    if pred:

encountered:

*Assert on "base_input.type() == input.type()" failed: Divergent data found*"""
        ),
    ):
        pipe = non_scalar_condition()
        pipe.run()


def test_operator_exception_arg_input_placement():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        batch_gpu = fn.copy(1.0).gpu() + 0
        return fn.random.coin_flip(probability=batch_gpu)

    with assert_raises(
        ValueError,
        glob=(
            "Invalid device \"gpu\" for argument 'probability' of operator"
            " 'nvidia.dali.fn.random.coin_flip'.*"
        ),
    ):
        p = pipe()
        p.run()


def test_operator_invalid_input_constant_promotion():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe0():
        return fn.zeros([1, 100])

    @pipeline_def(batch_size=4, num_threads=4, device_id=0)
    def pipe1():
        return fn.random.uniform([1, 100], 42)

    for fn_name, min_args, max_args, num_args, p_def in [
        ("fn.zeros", 0, 0, 1, pipe0),
        ("fn.random.uniform", 0, 1, 2, pipe1),
    ]:
        with assert_raises(
            ValueError,
            glob=(
                f"Operator nvidia.dali.{fn_name} expects from {min_args} "
                f"to {max_args} inputs, but received {num_args}."
            ),
        ):
            p = p_def()
            p.run()
