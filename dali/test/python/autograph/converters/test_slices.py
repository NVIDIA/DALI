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
"""Tests for slices module."""

from nvidia.dali._autograph.converters import directives as directives_converter
from nvidia.dali._autograph.converters import slices
from nvidia.dali._autograph.core import converter_testing


class SliceTest(converter_testing.TestCase):
    def test_index_access(self):
        def f(l):
            return l[1]

        tr = self.transform(f, (directives_converter, slices))

        tl = [1, 2]
        y = tr(tl)
        self.assertEqual(2, y)

    def test_index_access_multiple_definitions(self):
        def f(l):
            if l:
                l = []
            return l[1]

        self.transform(f, (directives_converter, slices))
