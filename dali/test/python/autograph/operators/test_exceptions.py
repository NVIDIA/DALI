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
"""Tests for exceptions module."""

import unittest

from nvidia.dali._autograph.operators import exceptions


class ExceptionsTest(unittest.TestCase):
    def test_assert_python_untriggered(self):
        side_effect_trace = []

        def expression_with_side_effects():
            side_effect_trace.append(object())
            return "test message"

        exceptions.assert_stmt(True, expression_with_side_effects)

        self.assertListEqual(side_effect_trace, [])

    def test_assert_python_triggered(self):
        if not __debug__:
            # Python assertions only be tested when in debug mode.
            return

        side_effect_trace = []
        tracer = object()

        def expression_with_side_effects():
            side_effect_trace.append(tracer)
            return "test message"

        with self.assertRaisesRegex(AssertionError, "test message"):
            exceptions.assert_stmt(False, expression_with_side_effects)
        self.assertListEqual(side_effect_trace, [tracer])
