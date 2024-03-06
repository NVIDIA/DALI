# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def, fn
from test_utils import load_test_operator_plugin
from nose_utils import assert_raises
from nose2.tools import params


def setUpModule():
    load_test_operator_plugin()


python_errors = [RuntimeError, IndexError, TypeError, ValueError, StopIteration]


@params(*python_errors)
def test_python_error_propagation(error):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type=error.__name__)

    with assert_raises(
        error,
        glob=(
            """Critical error in pipeline:
Error when executing CPU operator `fn.throw_exception`,
which was used in the pipeline definition with the following traceback:
  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type=error.__name__)
encountered:
Test message
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.build()
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
Error when executing CPU operator `fn.throw_exception`,
which was used in the pipeline definition with the following traceback:
  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type=error_name)
encountered:
Test message
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.build()
        p.run()


def test_error_propagation_ellipsis():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type="std::string")

    with assert_raises(
        RuntimeError,
        glob=(
            """Critical error in pipeline:
Error when executing CPU operator `fn.throw_exception`,
which was used in the pipeline definition with the following traceback:
  File "*test_operator_exception.py", line *, in pipe
    return fn.throw_exception(exception_type="std::string")
encountered:
Unknown critical error.
Current pipeline object is no longer valid."""
        ),
    ):
        p = pipe()
        p.build()
        p.run()
