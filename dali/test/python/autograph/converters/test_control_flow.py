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
"""Tests for control_flow module."""

import collections

from nvidia.dali._autograph.converters import break_statements
from nvidia.dali._autograph.converters import continue_statements
from nvidia.dali._autograph.converters import control_flow
from nvidia.dali._autograph.core import converter_testing
from nvidia.dali._autograph.utils.all_utils import custom_constant

for_unaffected_global = None
for_mixed_globals_nonglobals = None
for_test_global_local = None


class ControlFlowTestBase(converter_testing.TestCase):
    def assertValuesEqual(self, actual, expected):
        self.assertEqual(actual, expected)

    def assertTransformedResult(self, f, inputs, expected):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        tr = self.transform(f, control_flow)
        returns = tr(*inputs)
        self.assertValuesEqual(returns, expected)


class NestedControlFlowTest(ControlFlowTestBase):
    def test_basic(self):
        def f(n):
            i = 0
            j = 0
            s = 0
            while i < n:
                while j < i:
                    j += 3
                u = i + j  # 'u' is not defined within the inner loop
                s += u
                i += 1
                j = 0
            return s, i, j, n

        self.assertTransformedResult(f, custom_constant(5), (25, 5, 0, 5))

    def test_mixed_globals_nonglobals(self):
        def f(n):
            global for_mixed_globals_nonglobals
            i = 0
            j = 0
            for_mixed_globals_nonglobals = 0
            while i < n:
                while j < i:
                    j += 3
                u = i + j  # 'u' is not defined within the inner loop
                for_mixed_globals_nonglobals += u
                i += 1
                j = 0
            return for_mixed_globals_nonglobals, i, j, n

        self.assertTransformedResult(f, custom_constant(5), (25, 5, 0, 5))

    def test_composite_state_complex(self):
        class TestClassX(object):
            def __init__(self, x):
                self.x = x

        class TestClassY(object):
            def __init__(self, y):
                self.y = y

        def f(n):
            tc = TestClassX(TestClassY({"z": TestClassX(n)}))
            if n > 0:
                while n > 0:
                    if n < 2:
                        tc.x.y["z"].x += 1
                    n -= 1
            return n, tc

        tr = self.transform(f, control_flow)

        n, tc = tr(custom_constant(5))
        self.assertValuesEqual((n, tc.x.y["z"].x), (0, 6))


class WhileStatementTest(ControlFlowTestBase):
    def test_basic(self):
        def f(n):
            i = 0
            s = 0
            while i < n:
                s += i
                i += 1
            return s, i, n

        self.assertTransformedResult(f, custom_constant(5), (10, 5, 5))

    def test_single_output(self):
        def f(n):
            while n > 0:
                n -= 1
            return n

        self.assertTransformedResult(f, custom_constant(5), 0)

    def test_composite_state_attr(self):
        class TestClass(object):
            def __init__(self):
                self.x = custom_constant(3)

        def f(n):
            tc = TestClass()
            while n > 0:
                tc.x += 1
                n -= 1
            return n

        self.assertTransformedResult(f, custom_constant(5), 0)

    def test_composite_state_slice(self):
        def f(n):
            d = {"a": n}
            k = "a"
            while n > 0:
                d[k] += 1
                n -= 1
            return d[k], n

        self.assertTransformedResult(f, custom_constant(5), (10, 0))

    def test_composite_state_literal_slice(self):
        def f(n):
            d = {"a": n}
            while n > 0:
                d["a"] += 1
                n -= 1
            return d["a"], n

        self.assertTransformedResult(f, custom_constant(5), (10, 0))

    def test_local_composite_attr(self):
        class TestClass(object):
            def __init__(self):
                self.x = custom_constant(3)

        def f(n):
            while n > 0:
                tc = TestClass()
                tc.x = tc.x
                n -= 1
            return n

        self.assertTransformedResult(f, custom_constant(5), 0)

    def test_local_composite_slice(self):
        def f(n):
            while n > 0:
                d = {"x": n}
                k = "x"
                d[k] = d[k]
                n -= 1
            return n

        self.assertTransformedResult(f, custom_constant(5), 0)

    def test_local_composite_literal_slice(self):
        def f(n):
            while n > 0:
                d = {"x": n}
                d["x"] = d["x"]
                n -= 1
            return n

        self.assertTransformedResult(f, custom_constant(5), 0)

    def test_non_tensor_state(self):
        # This class is ok to be in a tf.while's state.
        class TestClass(collections.namedtuple("TestClass", ("x"))):
            pass

        def f(n):
            tc = TestClass([custom_constant(0)])
            while n > 0:
                tc = TestClass([custom_constant(3)])
                tc.x[0] = tc.x[0] + 1
                n -= 1
            return tc.x[0]

        self.assertTransformedResult(f, custom_constant(5), 4)


class IfStatementTest(ControlFlowTestBase):
    def test_basic(self):
        def f(n):
            a = 0
            b = 0
            if n > 0:
                a = -n
            else:
                b = 2 * n
            return a, b

        self.assertTransformedResult(f, custom_constant(1), (-1, 0))
        self.assertTransformedResult(f, custom_constant(-1), (0, -2))

    def test_complex_outputs(self):
        class TestClass(object):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def f(n, obj):
            obj.a = 0
            obj.b = 0
            if n > 0:
                obj.a = -n
            else:
                obj.b = 2 * n
            return obj

        tr = self.transform(f, control_flow)

        res_obj = tr(custom_constant(1), TestClass(0, 0))
        self.assertValuesEqual((res_obj.a, res_obj.b), (-1, 0))
        res_obj = tr(custom_constant(-1), TestClass(0, 0))
        self.assertValuesEqual((res_obj.a, res_obj.b), (0, -2))

    def test_single_output(self):
        def f(n):
            if n > 0:
                n = -n
            return n

        self.assertTransformedResult(f, custom_constant(1), -1)

    def test_unbalanced(self):
        def f(n):
            if n > 0:
                n = 3
            return n

        self.assertTransformedResult(f, custom_constant(2), 3)
        self.assertTransformedResult(f, custom_constant(-3), -3)

    def test_unbalanced_raising(self):
        def f(n):
            if n > 0:
                n = n + 1
                raise ValueError()
            return n

        self.assertTransformedResult(f, -3, -3)

        tr = self.transform(f, control_flow)

        with self.assertRaises(ValueError):
            tr(1)

    def test_local_var(self):
        def f(n):
            if n > 0:
                b = 4
                n = b + 1
            return n

        self.assertTransformedResult(f, custom_constant(1), 5)
        self.assertTransformedResult(f, custom_constant(-1), -1)

    def test_local_remains_local(self):
        def f(n):
            if n > 0:
                b = 4
                n = b + 1
            return n

        self.assertTransformedResult(f, custom_constant(1), 5)
        self.assertTransformedResult(f, custom_constant(-1), -1)

    def test_global_local(self):
        def f(n):
            if n > 0:
                global for_test_global_local
                if for_test_global_local is None:
                    for_test_global_local = 1
                else:
                    for_test_global_local += 1
                n += for_test_global_local
            return n

        tr = self.transform(f, control_flow)
        assert for_test_global_local is None
        self.assertEqual(tr(1), 2)
        self.assertEqual(for_test_global_local, 1)

    def test_no_outputs(self):
        def f(n):
            if n > 0:
                b = 4  # pylint:disable=unused-variable # noqa: F841
            return n

        self.assertTransformedResult(f, custom_constant(1), 1)
        self.assertTransformedResult(f, custom_constant(-1), -1)

    def test_created_outputs(self):
        def f(i):
            if i == 0:
                result = i - 1
            else:
                result = i + 1
            return result

        self.assertTransformedResult(f, 0, -1)
        self.assertTransformedResult(f, 1, 2)

    def test_created_loop_local_outputs(self):
        def f(n, x):
            for i in n:
                if i == 0:
                    result = i - 1
                else:
                    result = i + 1
                if result > 0:
                    x += 1
            return x

        self.assertTransformedResult(f, (range(5), 10), 14)

    def test_created_loop_variable(self):
        def f(n, x):
            for i in n:
                if i == 0:
                    result = i - 1
                if i > 0:  # Using the result from previous iteration.
                    if result < 0:
                        x += 1
            return x

        self.assertTransformedResult(f, (range(5), 10), 14)

    def test_unaffected_global(self):
        global for_unaffected_global
        for_unaffected_global = 3

        def f(i):
            global for_unaffected_global
            if i == 0:
                for_unaffected_global = i - 1
            return for_unaffected_global

        self.assertTransformedResult(f, 1, 3)
        self.assertTransformedResult(f, 0, -1)
        self.assertEqual(for_unaffected_global, -1)

    def test_unaffected_nonlocal(self):
        def f(i):
            def inner_fn():
                nonlocal n
                if i == 0:
                    n = i - 1

            n = 3
            inner_fn()
            return n

        self.assertTransformedResult(f, 1, 3)
        self.assertTransformedResult(f, 0, -1)

    def test_output_defined_in_prior_except(self):
        def f(i):
            try:
                raise ValueError()
            except ValueError:
                x = 1
            if i == 0:
                x = i - 1
            return x

        self.assertTransformedResult(f, 1, 1)
        self.assertTransformedResult(f, 0, -1)

    def test_unbalanced_multiple_composites(self):
        class Foo(object):
            def __init__(self):
                self.b = 2
                self.c = 3

        def f(x, condition):
            z = 5
            if condition:
                x.b = 7
                x.c = 11
                z = 13

            return x.b, x.c, z

        self.assertTransformedResult(f, (Foo(), custom_constant(True)), (7, 11, 13))
        self.assertTransformedResult(f, (Foo(), custom_constant(False)), (2, 3, 5))

    def test_unbalanced_composite(self):
        class Foo(object):
            def __init__(self):
                self.b = 2

        def f(x, condition):
            z = 5
            if condition:
                x.b = 7
                z = 13

            return x.b, z

        self.assertTransformedResult(f, (Foo(), custom_constant(True)), (7, 13))
        self.assertTransformedResult(f, (Foo(), custom_constant(False)), (2, 5))


class ForStatementTest(ControlFlowTestBase):
    def test_basic(self):
        def f(l):
            s1 = 0
            s2 = 0
            for e in l:
                s1 += e
                s2 += e * e
            return s1, s2

        self.assertTransformedResult(f, custom_constant([1, 3]), (4, 10))
        empty_vector = custom_constant([], shape=(0,), dtype=int)
        self.assertTransformedResult(f, empty_vector, (0, 0))

    def test_single_output(self):
        def f(l):
            s = 0
            for e in l:
                s += e
            return s

        self.assertTransformedResult(f, custom_constant([1, 3]), 4)
        empty_vector = custom_constant([], shape=(0,), dtype=int)
        self.assertTransformedResult(f, empty_vector, 0)

    def test_iterated_expression(self):
        eval_count = [0]

        def count_evals(x):
            eval_count[0] += 1
            return x

        def f(n):
            s = 0
            for e in count_evals(range(n)):
                s += e
            return s

        tr = self.transform(f, control_flow)

        self.assertEqual(tr(5), 10)
        self.assertEqual(eval_count[0], 1)

    def test_tuple_unpacking(self):
        def f(x_list):
            z = custom_constant(0)  # pylint:disable=undefined-variable # noqa: F821
            for i, x in enumerate(x_list):
                z = z + x + i
            return z

        self.assertTransformedResult(f, [3, 3], 7)

    def test_with_comprehension_in_body(self):
        def f(l, n):
            s = custom_constant(list(range(n)))
            for _ in l:
                s += custom_constant([a for a in range(n)])
            return s

        self.assertTransformedResult(f, (custom_constant([1, 2, 3]), 5), list(range(5)) * 4)


class AdvancedControlFlowTest(ControlFlowTestBase):
    def assertTransformedEquivalent(self, f, *inputs):
        tr = self.transform(f, (break_statements, continue_statements, control_flow))
        self.assertEqual(f(*inputs), tr(*inputs))

    def test_while_with_else(self):
        def f(x):
            while x > 2:
                x /= 2
            else:
                x += 1
            return x

        self.assertTransformedEquivalent(f, 4)
        self.assertTransformedEquivalent(f, 2)

    def test_while_with_else_and_break(self):
        def f(cond1):
            x = 8
            while x > 2:
                x /= 2
                if cond1:
                    break
            else:
                x += 1
            return x

        self.assertTransformedEquivalent(f, True)
        self.assertTransformedEquivalent(f, False)

    def test_for_with_else(self):
        def f(l):
            res = 0
            for x in l:
                res += x
            else:
                res += 1
            return res

        self.assertTransformedEquivalent(f, [])
        self.assertTransformedEquivalent(f, [1, 2])

    def test_for_with_else_and_break(self):
        def f(flag):
            l = [1, 2, 3]
            res = 0
            for x in l:
                res += x
                if flag:
                    break
            else:
                res += 1
            return res

        self.assertTransformedEquivalent(f, True)
        self.assertTransformedEquivalent(f, False)
