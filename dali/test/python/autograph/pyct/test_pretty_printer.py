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
"""Tests for pretty_printer module."""

import ast
import textwrap
import unittest

from nvidia.dali._autograph.pyct import pretty_printer


class PrettyPrinterTest(unittest.TestCase):
    def test_unicode_bytes(self):
        source = textwrap.dedent("""
    def f():
      return b'b', u'u', 'depends_py2_py3'
    """)
        node = ast.parse(source)
        self.assertIsNotNone(pretty_printer.fmt(node))

    def test_format(self):
        node = ast.parse("def f(a):\n    return a + 1\n").body[0]
        # Just checking for functionality, the color control characters make it
        # difficult to inspect the result.
        self.assertIsNotNone(pretty_printer.fmt(node))
