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
"""Tests for functions module."""

from nvidia.dali._autograph.converters import functions
from nvidia.dali._autograph.converters import return_statements
from nvidia.dali._autograph.core import ag_ctx
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.core import converter_testing
from nvidia.dali._autograph.impl import api

CONSTANT = 1


class FunctionTransformer(converter_testing.TestCase):
    def test_basic(self):
        def f(l):
            """Docstring."""
            a = 1
            l += a
            return l

        tr = self.transform(f, functions)

        result = tr(CONSTANT)
        self.assertEqual(2, result)
        self.assertEqual("Docstring.", tr.__doc__)

    def test_multiline_docstring(self):
        def f():
            """First sentence.

            Second sentence.

            Returns:
              Something.
            """
            return CONSTANT

        tr = self.transform(f, functions)

        result = tr()
        self.assertEqual(CONSTANT, result)
        self.assertIn("First sentence.", tr.__doc__)
        self.assertIn("Second sentence.", tr.__doc__)

    def test_nested_functions(self):
        def f(l):
            def inner_fn(i):
                return i + 1

            l += 1
            return l, inner_fn(l)

        tr = self.transform(f, (functions, return_statements))

        first, second = tr(CONSTANT)

    def test_conversion_context_preserves_in_inner_functions(self):
        def inner_fn_callee():
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.DISABLED)

        def f():
            def inner_fn():
                inner_fn_callee()

            with ag_ctx.ControlStatusCtx(
                ag_ctx.Status.DISABLED, converter.ConversionOptions(recursive=True)
            ):
                inner_fn()

        tr = self.transform(f, functions)

        tr()

    def test_method(self):
        class TestClass(object):
            def f(self, l):
                def inner_fn(i):
                    return i + 1

                l += 1
                return l, inner_fn(l)

        tr = self.transform(TestClass.f, (functions, return_statements))

        first, second = tr(TestClass(), CONSTANT)

    def test_lambda_in_return_value(self):
        def f():
            return lambda x: x + 1

        tr = self.transform(f, functions)

        result_l = tr()
        self.assertTrue(api.is_autograph_artifact(result_l))
