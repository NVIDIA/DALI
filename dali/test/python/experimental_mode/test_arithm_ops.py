# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nose_utils import assert_raises, attr
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
            a = ndd.tensor(ref_a, device=device)
            b = ndd.tensor(ref_b, device=device)
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
        x = ndd.tensor(ref_x, device=device)
        y = apply_un_op(op, x)
        ref_y = apply_un_op(op, ref_x)
        if not np.array_equal(asnumpy(y), ref_y):
            msg = f"{ref_x} {op} = \n{asnumpy(y)}\n!=\n{ref_y}"
            raise AssertionError(msg)


@params(*itertools.product(["gpu", "cpu"], binary_ops, (None, 4)))
def test_binary_scalars(device: str, op: str, batch_size: int | None):
    tensors = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1], [2], [3]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
    ]
    scalars = [3, [4, 5, 6]]

    for tensor, scalar in itertools.product(tensors, scalars):
        if op == "/":
            tensor = tensor.astype(np.float32)

        if batch_size is None:
            x = ndd.as_tensor(tensor, device=device)
        else:
            x = ndd.Batch.broadcast(tensor, batch_size=batch_size, device=device)

        result = ndd.as_tensor(apply_bin_op(op, x, scalar))
        result_rev = ndd.as_tensor(apply_bin_op(op, scalar, x))
        ref = apply_bin_op(op, tensor, scalar)
        ref_rev = apply_bin_op(op, scalar, tensor)

        # np.allclose supports broadcasting
        if not np.allclose(result.cpu(), ref):
            msg = f"{tensor} {op} {scalar} = \n{result}\n!=\n{ref}"
            raise AssertionError(msg)

        if not np.allclose(result_rev.cpu(), ref_rev):
            msg = f"{scalar} {op} {tensor} = \n{result_rev}\n!=\n{ref_rev}"
            raise AssertionError(msg)


@attr("pytorch")
@params(*binary_ops)
def test_binary_pytorch_gpu(op: str):
    import torch

    a = torch.tensor([1, 2, 3], device="cuda")
    b = ndd.as_tensor(a)

    result = apply_bin_op(op, a, b)
    result_rev = apply_bin_op(op, b, a)
    expected = apply_bin_op(op, a, a)
    np.testing.assert_array_equal(result.cpu(), expected.cpu())
    np.testing.assert_array_equal(expected.cpu(), result_rev.cpu())


@params(*binary_ops)
def test_incompatible_devices(op: str):
    a = ndd.tensor([1, 2, 3], device="cpu")
    b = ndd.tensor([4, 5, 6], device="gpu")

    with assert_raises(ValueError, regex="[CG]PU and [CG]PU"):
        apply_bin_op(op, a, b)
    with assert_raises(ValueError, regex="[CG]PU and [CG]PU"):
        apply_bin_op(op, b, a)


@attr("pytorch")
@params(*binary_ops)
def test_binary_pytorch_incompatible(op: str):
    import torch

    devices = [
        ("cpu", "gpu"),
        ("cuda", "cpu"),
    ]

    for torch_device, ndd_device in devices:
        a = torch.tensor([1, 2, 3], device=torch_device)
        b = ndd.tensor([1, 2, 3], device=ndd_device)

        with assert_raises(ValueError, regex="[CG]PU and [CG]PU"):
            apply_bin_op(op, a, b)
        with assert_raises(ValueError, regex="[CG]PU and [CG]PU"):
            apply_bin_op(op, b, a)
