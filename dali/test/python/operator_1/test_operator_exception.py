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


errors = [RuntimeError, IndexError, TypeError, ValueError, StopIteration]


@params(*errors)
def test_error_propagation(error):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.throw_exception(exception_type=error.__name__)

    with assert_raises(
        error, glob="Error when executing CPU operator ThrowException encountered:\nTest message"
    ):
        p = pipe()
        p.build()
        p.run()
