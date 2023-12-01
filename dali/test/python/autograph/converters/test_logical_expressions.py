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
"""Tests for logical_expressions module."""

from nvidia.dali._autograph.converters import logical_expressions
from nvidia.dali._autograph.core import converter_testing


class LogicalExpressionTest(converter_testing.TestCase):
    def test_equals(self):
        def f(a, b):
            return a == b

        tr = self.transform(f, logical_expressions)

        self.assertTrue(tr(1, 1))
        self.assertFalse(tr(1, 2))

    def test_bool_ops(self):
        def f(a, b, c):
            return (a or b) and (a or b or c) and not c

        tr = self.transform(f, logical_expressions)

        self.assertTrue(tr(True, False, False))
        self.assertFalse(tr(True, False, True))

    def test_comparison(self):
        def f(a, b, c, d):
            return a < b == c > d

        tr = self.transform(f, logical_expressions)

        # Note: having just the first constant a tensor tests that the
        # operations execute in the correct order. If anything other than
        # a < b executed first, the result would be a Python scalar and not a
        # Tensor. This is valid as long as the dispatch is automatic based on
        # type.
        self.assertTrue(tr(1, 2, 2, 1))
        self.assertFalse(tr(1, 2, 2, 3))

    def test_default_ops(self):
        def f(a, b):
            return a in b

        tr = self.transform(f, logical_expressions)

        self.assertTrue(tr("a", ("a",)))

    def test_unary_ops(self):
        def f(a):
            return ~a, -a, +a

        tr = self.transform(f, logical_expressions)

        self.assertEqual(tr(1), (-2, -1, 1))
