# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import unittest

from nvidia.dali._autograph.operators import slices
from nvidia.dali._autograph.utils.all_utils import custom_constant


class SlicesTest(unittest.TestCase):
    def test_set_item_tensor_list(self):
        initial_list = custom_constant([[1, 2], [3, 4]])
        l = slices.set_item(initial_list, 0, [5, 6])
        self.assertEqual(l, [[5, 6], [3, 4]])

    def test_get_item_tensor_list(self):
        initial_list = custom_constant([[1, 2], [3, 4]])
        t = slices.get_item(initial_list, 1, slices.GetItemOpts(None))
        self.assertEqual(t, [3, 4])

    def test_get_item_tensor_string(self):
        initial_str = custom_constant("abcd")
        t = slices.get_item(initial_str, 1, slices.GetItemOpts(None))
        self.assertEqual(t, "b")

        initial_list_str = custom_constant(["abcd", "bcde"])
        t = slices.get_item(initial_list_str, 1, slices.GetItemOpts(None))
        self.assertEqual(t, "bcde")
