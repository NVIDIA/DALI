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
"""Tests for lists module."""

from nvidia.dali._autograph.converters import directives as directives_converter
from nvidia.dali._autograph.converters import lists
from nvidia.dali._autograph.core import converter_testing
from nvidia.dali._autograph.utils import hooks


class TestList(list):
    pass


class OperatorList(hooks.OperatorBase):
    def is_test_list(self, iterable):
        return isinstance(iterable, TestList)

    def detect_overload_list_new(self, iterable):
        return True

    def list_new(self, iterable):
        return TestList(iterable)

    def detect_overload_list_append(self, list_):
        return self.is_test_list(list_)

    def list_append(self, list_, x):
        list_.append(x)
        list_ = TestList(list_)
        return TestList(list_)

    def detect_overload_list_pop(self, list_):
        return self.is_test_list(list_)

    def list_pop(self, list_, i):
        if i is None:
            x = list_.pop()
        else:
            x = list_.pop(i)
        return list_, x


class ListTest(converter_testing.TestCase):
    def test_empty_list(self):
        def f():
            return []

        tr = self.transform(f, lists, operator_overload=OperatorList())

        tl = tr()
        # Empty tensor lists cannot be evaluated or stacked.
        self.assertIsInstance(tl, TestList)

    def test_initialized_list(self):
        def f():
            return [1, 2, 3]

        tr = self.transform(f, lists, operator_overload=OperatorList())
        tl = tr()

        self.assertIsInstance(tl, TestList)
        self.assertEqual(tl, [1, 2, 3])

    def test_list_append(self):
        def f():
            l = TestList([1])
            l.append(2)
            l.append(3)
            return l

        tr = self.transform(f, lists, operator_overload=OperatorList())

        tl = tr()

        self.assertIsInstance(tl, TestList)
        self.assertEqual(tl, [1, 2, 3])

    def test_list_pop(self):
        def f():
            l = TestList([1, 2, 3])
            s = l.pop()
            return s, l

        tr = self.transform(f, (directives_converter, lists), operator_overload=OperatorList())

        ts, tl = tr()

        self.assertIsInstance(tl, TestList)
        self.assertEqual(tl, [1, 2])
        self.assertEqual(ts, 3)

    def test_double_list_pop(self):
        def f(l):
            s = l.pop().pop()
            return s, l

        tr = self.transform(f, lists, operator_overload=OperatorList())

        test_input = [1, 2, [1, 2, 3]]
        # TODO(mdan): Pass a list of lists of tensor when we fully support that.
        # For now, we just pass a regular Python list of lists just to verify that
        # the two pop calls are sequenced properly.
        s, tl = tr(test_input)

        self.assertIsInstance(tl, list)
        self.assertEqual(s, 3)

    # TODO(klecki): Revert the stack test
    # def test_list_stack(self):

    #   def f():
    #     l = [1, 2, 3]
    #     return array_ops.stack(l)

    #   tr = self.transform(f, lists, operator_overload=OperatorList())

    #   self.assertAllEqual(self.evaluate(tr()), [1, 2, 3])
