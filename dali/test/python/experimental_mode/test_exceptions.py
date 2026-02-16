# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import functools
from collections.abc import Callable

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose_utils import raises


def exception_tester(function: Callable[[], None]):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        initial_message = None
        expected_exception = None
        try:
            with ndd.EvalMode.sync_cpu:
                function()
        except Exception as exception:
            initial_message = str(exception)
            expected_exception = exception

        assert initial_message is not None and expected_exception is not None

        for eval_mode in (ndd.EvalMode.deferred, ndd.EvalMode.eager):
            with eval_mode:
                try:
                    function()
                except ndd._exceptions.DisplacedEvaluationError as exception:
                    cause = exception.__cause__
                    assert type(cause) is type(expected_exception)
                    assert str(cause) == str(expected_exception)
                else:
                    assert False

    return wrapper


@exception_tester
def test_bad_rank():
    img = ndd.zeros(shape=(100, 100))
    ndd.resize(img).evaluate()


@exception_tester
def test_bad_dtype():
    img = ndd.zeros(shape=(100, 100, 3), dtype=ndd.bool).gpu()
    ndd.resize(img, size=(50, 50)).evaluate()


@exception_tester
def test_bad_crop_size():
    img = ndd.zeros(shape=(100, 100, 3))
    ndd.crop(img, crop=(200, 200)).evaluate()


@exception_tester
def test_ragged_batch_to_tensor():
    batch = ndd.batch([[1, 2], [3]])
    tensor = ndd.as_tensor(batch)
    tensor.evaluate()


@raises(ValueError)
def test_with_external():
    img = np.zeros((100, 100))
    # Here, img is an external tensor so async execution shouldn't be used
    ndd.resize(img)  # no need to evaluate here
