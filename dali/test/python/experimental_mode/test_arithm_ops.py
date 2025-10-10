# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as D
from nose2.tools import params
import numpy as np
import itertools
from test_tensor import asnumpy


def apply_bin_op(op, a, b):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    elif op == "//":
        return a // b
    elif op == "**":
        return a**b
    elif op == "&":
        return a & b
    elif op == "|":
        return a | b
    elif op == "^":
        return a ^ b
    elif op == "==":
        return a == b
    elif op == "!=":
        return a != b
    elif op == "<":
        return a < b
    elif op == "<=":
        return a <= b
    elif op == ">":
        return a > b
    elif op == ">=":
        return a >= b


def apply_un_op(op, a):
    if op == "+":
        return +a
    elif op == "-":
        return -a
    elif op == "~":
        return ~a


binary_ops = ["+", "-", "*", "/", "//", "**", "&", "|", "^", "==", "!=", "<", "<=", ">", ">="]
unary_ops = ["+", "-"]  # TODO(michalz): ~ missing in DALI - fix!


@params(*itertools.product(["cpu", "gpu"], binary_ops))
def test_binary_ops(device, op):
    values = [
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[2], [3]])),
        (np.array([[1], [2], [3]]), 5),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2, 3])),
    ]

    for va, vb in values:
        for ref_a, ref_b in [(va, vb), (vb, va)]:
            if op == "/":
                ref_a = np.float32(ref_a)
                ref_b = np.float32(ref_b)
            a = D.tensor(ref_a, device=device)
            b = D.tensor(ref_b, device=device)
            ab = apply_bin_op(op, a, b)
            result_numpy = asnumpy(ab)
            ref_ab = apply_bin_op(op, ref_a, ref_b)
            if not np.array_equal(result_numpy, ref_ab):
                msg = f"{ref_a} {op} {ref_b} = \n{result_numpy}\n!=\n{ref_ab}"
                raise AssertionError(msg)


@params(*itertools.product(["cpu", "gpu"], unary_ops))
def test_unary_ops(device, op):
    values = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1], [2], [3]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
    ]

    for ref_x in values:
        x = D.tensor(ref_x, device=device)
        y = apply_un_op(op, x)
        ref_y = apply_un_op(op, ref_x)
        if not np.array_equal(asnumpy(y), ref_y):
            msg = f"{ref_x} {op} = \n{asnumpy(y)}\n!=\n{ref_y}"
            raise AssertionError(msg)
