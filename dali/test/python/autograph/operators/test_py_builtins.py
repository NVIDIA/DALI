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
"""Tests for py_builtins module."""

import unittest

from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.core import function_wrappers
from nvidia.dali._autograph.operators import py_builtins


class TestBase(object):
    def overridden_method(self, x):
        return x + 20


class PyBuiltinsTest(unittest.TestCase):
    def test_abs(self):
        self.assertEqual(py_builtins.abs_(-1), 1)

    def test_float(self):
        self.assertEqual(py_builtins.float_(10), 10.0)
        self.assertEqual(py_builtins.float_("10.0"), 10.0)

    def test_int(self):
        self.assertEqual(py_builtins.int_(10.0), 10)
        self.assertEqual(py_builtins.int_("11", 2), 3)

    def test_int_unsupported_base(self):
        t = 1.0
        with self.assertRaises(TypeError):
            py_builtins.int_(t, 2)

    def test_len(self):
        self.assertEqual(py_builtins.len_([1, 2, 3]), 3)

    def test_len_scalar(self):
        with self.assertRaises(TypeError):
            py_builtins.len_(1)

    def test_max(self):
        self.assertEqual(py_builtins.max_([1, 3, 2]), 3)
        self.assertEqual(py_builtins.max_(0, 2, 1), 2)

    def test_min(self):
        self.assertEqual(py_builtins.min_([2, 1, 3]), 1)
        self.assertEqual(py_builtins.min_(2, 0, 1), 0)

    def test_range(self):
        self.assertListEqual(list(py_builtins.range_(3)), [0, 1, 2])
        self.assertListEqual(list(py_builtins.range_(1, 3)), [1, 2])
        self.assertListEqual(list(py_builtins.range_(2, 0, -1)), [2, 1])

    def test_enumerate(self):
        self.assertListEqual(list(py_builtins.enumerate_([3, 2, 1])), [(0, 3), (1, 2), (2, 1)])
        self.assertListEqual(list(py_builtins.enumerate_([3, 2, 1], 5)), [(5, 3), (6, 2), (7, 1)])
        self.assertListEqual(list(py_builtins.enumerate_([-8], -3)), [(-3, -8)])

    def test_zip(self):
        self.assertListEqual(list(py_builtins.zip_([3, 2, 1], [1, 2, 3])), [(3, 1), (2, 2), (1, 3)])
        self.assertListEqual(list(py_builtins.zip_([4, 5, 6], [-1, -2])), [(4, -1), (5, -2)])

    def test_map(self):
        def increment(x):
            return x + 1

        add_list = lambda x, y: x + y
        self.assertListEqual(list(py_builtins.map_(increment, [4, 5, 6])), [5, 6, 7])
        self.assertListEqual(list(py_builtins.map_(add_list, [3, 2, 1], [-1, -2, -3])), [2, 0, -2])

    def test_next_normal(self):
        iterator = iter([1, 2, 3])
        self.assertEqual(py_builtins.next_(iterator), 1)
        self.assertEqual(py_builtins.next_(iterator), 2)
        self.assertEqual(py_builtins.next_(iterator), 3)
        with self.assertRaises(StopIteration):
            py_builtins.next_(iterator)
        self.assertEqual(py_builtins.next_(iterator, 4), 4)

    def _basic_function_scope(self):
        return function_wrappers.FunctionScope(
            "test_function_name",
            "test_scope",  # Note: this must match the name in the `with` statement.
            converter.ConversionOptions(),
        )

    def test_eval_in_original_context(self):
        def test_fn():
            l = 1  # pylint:disable=unused-variable # noqa: F841
            with self._basic_function_scope() as test_scope:
                return py_builtins.eval_in_original_context(eval, ("l",), test_scope)

        self.assertEqual(test_fn(), 1)

    def test_eval_in_original_context_inner_function(self):
        def test_fn():
            l = 1  # pylint:disable=unused-variable # noqa: F841
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    # Note: a user function without a top-level function scope should
                    # never be found in user code; it's only possible in generated code.
                    l = 2  # pylint:disable=unused-variable # noqa: F841
                    return py_builtins.eval_in_original_context(eval, ("l",), test_scope)

                return inner_fn()

        self.assertEqual(test_fn(), 2)

    def test_locals_in_original_context(self):
        def test_fn():
            l = 1  # pylint:disable=unused-variable # noqa: F841
            with self._basic_function_scope() as test_scope:
                return py_builtins.locals_in_original_context(test_scope)

        locs = test_fn()

        self.assertEqual(locs["l"], 1)

    def test_locals_in_original_context_inner_function(self):
        def test_fn():
            l = 1  # pylint:disable=unused-variable # noqa: F841
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    # Note: a user function without a top-level function scope should
                    # never be found in user code; it's only possible in generated code.
                    l = 2  # pylint:disable=unused-variable # noqa: F841
                    return py_builtins.locals_in_original_context(test_scope)

                return inner_fn()

        locs = test_fn()

        self.assertEqual(locs["l"], 2)

    def test_globals_in_original_context(self):
        def test_fn():
            with self._basic_function_scope() as test_scope:
                return py_builtins.globals_in_original_context(test_scope)

        globs = test_fn()

        self.assertIs(globs["TestBase"], TestBase)

    def test_globals_in_original_context_inner_function(self):
        def test_fn():
            with self._basic_function_scope() as test_scope:

                def inner_fn():
                    # Note: a user function without a top-level function scope should
                    # never be found in user code; it's only possible in generated code.
                    return py_builtins.globals_in_original_context(test_scope)

                return inner_fn()

        globs = test_fn()

        self.assertIs(globs["TestBase"], TestBase)

    def test_super_in_original_context_unary_call(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self):
                with test_case_self._basic_function_scope() as test_scope:
                    test_base_unbound = py_builtins.super_in_original_context(
                        super, (TestSubclass,), test_scope
                    )
                    test_base = test_base_unbound.__get__(self, TestSubclass)
                    return test_base.overridden_method(1)

        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_binary_call(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self):
                with test_case_self._basic_function_scope() as test_scope:
                    test_base = py_builtins.super_in_original_context(
                        super, (TestSubclass, self), test_scope
                    )
                    return test_base.overridden_method(1)

        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_niladic_call(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self):
                with test_case_self._basic_function_scope() as test_scope:
                    b = py_builtins.super_in_original_context(super, (), test_scope)
                    return b.overridden_method(1)

        tc = TestSubclass()
        self.assertEqual(tc.test_method(), 21)

    def test_super_in_original_context_caller_with_locals(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self, x):
                y = 7
                with test_case_self._basic_function_scope() as test_scope:
                    z = 7
                    return py_builtins.super_in_original_context(
                        super, (), test_scope
                    ).overridden_method(x + y - z)

        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_super_in_original_context_inner_function(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self, x):
                with test_case_self._basic_function_scope() as test_scope:
                    # Oddly, it's sufficient to use `self` in an inner function
                    # to gain access to __class__ in this scope.
                    # TODO(mdan): Is this true across implementations?
                    # Note: normally, it's illegal to use super() in inner functions (it
                    # throws an error), but the generated code may create them.
                    def inner_fn():
                        return py_builtins.super_in_original_context(
                            super, (), test_scope
                        ).overridden_method(x)

                    return inner_fn()

        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_super_in_original_context_inner_lambda(self):
        test_case_self = self

        class TestSubclass(TestBase):
            def overridden_method(self, x):
                test_case_self.fail("This should never be called.")

            def test_method(self, x):
                with test_case_self._basic_function_scope() as test_scope:
                    # Oddly, it's sufficient to use `self` in an inner function
                    # to gain access to __class__ in this scope.
                    # TODO(mdan): Is this true across implementations?
                    # Note: normally, it's illegal to use super() in inner functions (it
                    # throws an error), but the generated code may create them.
                    l = lambda: py_builtins.super_in_original_context(  # pylint:disable=g-long-lambda # noqa: E501
                        super, (), test_scope
                    ).overridden_method(
                        x
                    )
                    return l()

        tc = TestSubclass()
        self.assertEqual(tc.test_method(1), 21)

    def test_filter(self):
        self.assertListEqual(list(py_builtins.filter_(lambda x: x == "b", ["a", "b", "c"])), ["b"])
        self.assertListEqual(list(py_builtins.filter_(lambda x: x < 3, [3, 2, 1])), [2, 1])

    def test_any(self):
        self.assertEqual(py_builtins.any_([False, True, False]), True)
        self.assertEqual(py_builtins.any_([False, False, False]), False)

    def test_all(self):
        self.assertEqual(py_builtins.all_([False, True, False]), False)
        self.assertEqual(py_builtins.all_([True, True, True]), True)

    def test_sorted(self):
        self.assertListEqual(py_builtins.sorted_([2, 3, 1]), [1, 2, 3])
        self.assertListEqual(py_builtins.sorted_([2, 3, 1], key=lambda x: -x), [3, 2, 1])
        self.assertListEqual(py_builtins.sorted_([2, 3, 1], reverse=True), [3, 2, 1])
        self.assertListEqual(
            py_builtins.sorted_([2, 3, 1], key=lambda x: -x, reverse=True), [1, 2, 3]
        )
        self.assertEqual(
            py_builtins.sorted_([[4, 3], [2, 1]], key=lambda x: sum(x)), [[2, 1], [4, 3]]
        )
