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
"""Tests for data_structures module."""

import unittest

from nvidia.dali._autograph.operators import data_structures


class ListTest(unittest.TestCase):
    def test_append_python(self):
        l = []
        self.assertEqual(data_structures.list_append(l, 1), [1])
        self.assertEqual(data_structures.list_append(l, 2), [1, 2])

    def test_pop_python(self):
        l = [1, 2, 3]
        opts = data_structures.ListPopOpts(element_dtype=None, element_shape=())
        self.assertEqual(data_structures.list_pop(l, None, opts), ([1, 2], 3))
        self.assertEqual(data_structures.list_pop(l, None, opts), ([1], 2))
