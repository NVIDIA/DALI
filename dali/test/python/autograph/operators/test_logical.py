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
"""Tests for logical module."""

import unittest

from nvidia.dali._autograph.operators import logical


class LogicalOperatorsTest(unittest.TestCase):
    def assertNotCalled(self):
        self.fail("this should not be called")

    def test_and_python(self):
        self.assertTrue(logical.and_(lambda: True, lambda: True))
        self.assertTrue(logical.and_(lambda: [1], lambda: True))
        self.assertListEqual(logical.and_(lambda: True, lambda: [1]), [1])

        self.assertFalse(logical.and_(lambda: False, lambda: True))
        self.assertFalse(logical.and_(lambda: False, self.assertNotCalled))

    def test_or_python(self):
        self.assertFalse(logical.or_(lambda: False, lambda: False))
        self.assertFalse(logical.or_(lambda: [], lambda: False))
        self.assertListEqual(logical.or_(lambda: False, lambda: [1]), [1])

        self.assertTrue(logical.or_(lambda: False, lambda: True))
        self.assertTrue(logical.or_(lambda: True, self.assertNotCalled))

    def test_not_python(self):
        self.assertFalse(logical.not_(True))
        self.assertFalse(logical.not_([1]))
        self.assertTrue(logical.not_([]))
