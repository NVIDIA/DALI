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

# Unfortunately pylint has false positives when nonlocal is present.
# pylint:disable=unused-variable

import unittest

from nvidia.dali._autograph.operators import control_flow


class ForLoopTest(unittest.TestCase):
    def test_python(self):
        def body(i):
            nonlocal s
            s = s * 10 + i

        def set_state(loop_vars):
            nonlocal s
            (s,) = loop_vars

        s = 0
        control_flow.for_stmt(
            range(5),
            extra_test=lambda: True,
            body=body,
            get_state=lambda: (s,),
            set_state=set_state,
            symbol_names=("s",),
            opts={},
        )

        self.assertEqual(s, 1234)

    def test_python_generator_with_extra_test(self):
        def new_generator():
            for i in range(1, 5):
                yield i

        gen = new_generator()

        def run_loop():
            s = 0
            c = 0

            def body(i):
                nonlocal s, c
                s = s * 10 + i
                c += 1

            control_flow.for_stmt(
                gen,
                extra_test=lambda: c == 0,  # Break after first iteration
                body=body,
                get_state=None,
                set_state=None,
                symbol_names=("s", "c"),
                opts={},
            )
            return s, c

        self.assertEqual(run_loop(), (1, 1))
        self.assertEqual(run_loop(), (2, 1))
        self.assertEqual(run_loop(), (3, 1))

        self.assertEqual(next(gen), 4)

    def test_python_generator_with_extra_test_no_iterations(self):
        def new_generator():
            for i in range(5):
                yield i

        gen = new_generator()

        def run_loop():
            s = 0

            def body(i):
                nonlocal s
                s = s * 10 + i

            control_flow.for_stmt(
                gen,
                extra_test=lambda: False,  # Break before loop
                body=body,
                get_state=None,
                set_state=None,
                symbol_names=("s",),
                opts={},
            )
            return s

        self.assertEqual(run_loop(), 0)
        self.assertEqual(run_loop(), 0)

        self.assertEqual(next(gen), 0)


class WhileLoopTest(unittest.TestCase):
    def test_python(self):
        def body():
            nonlocal i, s
            s = s * 10 + i
            i += 1

        i = 0
        s = 0
        n = 5
        control_flow.while_stmt(
            test=lambda: i < n,
            body=body,
            get_state=None,
            set_state=None,
            symbol_names=("i", "s"),
            opts={},
        )

        self.assertEqual(s, 1234)

    def test_python_with_tensor_state(self):
        def body():
            nonlocal i, s
            s = s * 10 + i
            i += 1

        i = 0
        s = 0
        n = 5
        control_flow.while_stmt(
            test=lambda: i < n,
            body=body,
            get_state=None,
            set_state=None,
            symbol_names=("i", "s"),
            opts={},
        )

        self.assertEqual(i, 5)
        self.assertEqual(s, 1234)


class IfStmtTest(unittest.TestCase):
    def test_python(self):
        def test_fn(cond):
            def body():
                nonlocal i
                i = 1

            def orelse():
                nonlocal i
                i = -1

            i = None
            control_flow.if_stmt(
                cond=cond,
                body=body,
                orelse=orelse,
                get_state=None,
                set_state=None,
                symbol_names=("i",),
                nouts=1,
            )
            return i

        self.assertEqual(test_fn(True), 1)
        self.assertEqual(test_fn(False), -1)

    def test_python_multiple_returns(self):
        def test_fn(cond):
            def body():
                nonlocal i, j
                i = 1
                j = 2

            def orelse():
                nonlocal i, j
                i = -1
                j = -2

            i, j = None, None
            control_flow.if_stmt(
                cond=cond,
                body=body,
                orelse=orelse,
                get_state=None,
                set_state=None,
                symbol_names=("i", "j"),
                nouts=2,
            )
            return i, j

        self.assertEqual(test_fn(True), (1, 2))
        self.assertEqual(test_fn(False), (-1, -2))
