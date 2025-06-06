# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for asserts module."""

from nvidia.dali._autograph.converters import asserts
from nvidia.dali._autograph.converters import functions
from nvidia.dali._autograph.converters import return_statements
from nvidia.dali._autograph.core import converter_testing


class AssertsTest(converter_testing.TestCase):
    def test_basic(self):
        def f(a):
            assert a, "testmsg"
            return a

        tr = self.transform(f, (functions, asserts, return_statements))

        with self.assertRaisesRegex(AssertionError, "testmsg"):
            _ = tr(False)
