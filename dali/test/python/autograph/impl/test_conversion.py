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
"""Tests for conversion module."""

import types
import sys
import unittest

import six

from nvidia.dali._autograph import utils
from nvidia.dali._autograph.core import config
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.impl import api
from nvidia.dali._autograph.impl import conversion


class ConversionTest(unittest.TestCase):
    def _simple_program_ctx(self):
        return converter.ProgramContext(
            options=converter.ConversionOptions(recursive=True), autograph_module=api
        )

    def test_is_allowlisted(self):
        def test_fn():
            return 1

        self.assertFalse(conversion.is_allowlisted(test_fn))

    def test_is_allowlisted_callable_allowlisted_call(self):
        allowlisted_mod = types.ModuleType("test_allowlisted_call")
        sys.modules["test_allowlisted_call"] = allowlisted_mod
        config.CONVERSION_RULES = (
            config.DoNotConvert("test_allowlisted_call"),
        ) + config.CONVERSION_RULES

        class TestClass(object):
            def __call__(self):
                pass

            def allowlisted_method(self):
                pass

        TestClass.__module__ = "test_allowlisted_call"
        if six.PY2:
            TestClass.__call__.__func__.__module__ = "test_allowlisted_call"
        else:
            TestClass.__call__.__module__ = "test_allowlisted_call"

        class Subclass(TestClass):
            def converted_method(self):
                pass

        tc = Subclass()

        self.assertTrue(conversion.is_allowlisted(TestClass.__call__))
        self.assertTrue(conversion.is_allowlisted(tc))
        self.assertTrue(conversion.is_allowlisted(tc.__call__))
        self.assertTrue(conversion.is_allowlisted(tc.allowlisted_method))
        self.assertFalse(conversion.is_allowlisted(Subclass))
        self.assertFalse(conversion.is_allowlisted(tc.converted_method))
