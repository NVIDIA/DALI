# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
from nose2.tools import params
from nose_utils import assert_raises
from nvidia.dali import pipeline_def, fn
from nvidia.dali import types


def run(op):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe0():
        return op

    p = pipe0()
    return np.array(p.run()[0][0])


def run_with_layout(op):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe0():
        return op

    p = pipe0()
    out = p.run()[0]
    return np.array(out[0]), out.layout()


def test_zeros():
    sh = (2, 3)
    np.testing.assert_array_equal(run(fn.zeros(shape=sh)), np.zeros(shape=sh))


def test_zeros_like():
    sh = (2, 3)
    arr = np.ones(sh)
    np.testing.assert_array_almost_equal(run(fn.zeros_like(arr)), np.zeros_like(arr))


def test_ones():
    sh = (2, 3)
    np.testing.assert_array_almost_equal(run(fn.ones(shape=sh)), np.ones(shape=sh))


def test_ones_like():
    sh = (2, 3)
    arr = np.ones(sh)
    np.testing.assert_array_almost_equal(run(fn.ones_like(arr)), np.ones_like(arr))


def test_full_scalar():
    sh = (2, 3, 4)
    np.testing.assert_array_almost_equal(run(fn.full(1234, shape=sh)), np.full(sh, 1234))


def test_full_broadcast_simple():
    shape = (2, 3)
    data = np.array([1, 2, 3], dtype=np.int32)
    assert data.shape == (3,)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))

    shape = (2, 4)  # inner extent is a power of 2
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    assert data.shape == (4,)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))

    shape = (3, 2)
    data = np.array([[1], [2], [3]], dtype=np.int32)
    assert data.shape == (3, 1)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))


def test_full_broadcast_complex():
    shape = (2, 3, 4)
    # broadcast along middle dim
    data = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int32)
    assert data.shape == (2, 1, 4)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))

    # broadcast along outer and inner dim, keep middle
    shape = (2, 3, 4)
    data = np.array([[[1], [2], [3]]], dtype=np.int32)
    assert data.shape == (1, 3, 1)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))

    # broadcast along axes 0 and 2
    shape = (2, 3, 4, 3)
    data = np.array([[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]], dtype=np.int32)
    assert data.shape == (1, 3, 1, 3)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))

    # broadcast along axes 1 and 3
    shape = (2, 3, 4, 3)
    data = np.array([[[[1], [2], [3], [4]]], [[[5], [6], [7], [8]]]], dtype=np.int32)
    assert data.shape == (2, 1, 4, 1)
    np.testing.assert_array_almost_equal(run(fn.full(data, shape=shape)), np.full(shape, data))


def test_full_like():
    sh = (2, 3, 4)
    fill_value_sh = (3, 4)
    arr = np.random.uniform(size=sh)
    fill_value_arr = np.random.uniform(size=fill_value_sh)
    np.testing.assert_array_almost_equal(
        run(fn.full_like(arr, fill_value_arr)), np.full_like(arr, fill_value_arr)
    )


@params(
    (fn.zeros, np.zeros),
    (fn.ones, np.ones),
)
def test_const_layout(op, np_op):
    sh = (2, 3, 4)
    layout = "HWC"
    arr, out_layout = run_with_layout(op(shape=sh, layout=layout))
    np.testing.assert_array_equal(arr, np_op(shape=sh))
    assert out_layout == layout


@params(
    (fn.zeros_like, np.zeros_like),
    (fn.ones_like, np.ones_like),
)
def test_const_like_layout(op, np_op):
    sh = (2, 3, 4)
    layout = "HWC"
    arr = types.Constant(np.ones(sh), layout=layout)
    result, out_layout = run_with_layout(op(arr))
    np.testing.assert_array_equal(result, np_op(np.ones(sh)))
    assert out_layout == layout


@params(
    ((2, 3, 4), "HWC", 42),
    ((3, 5), "HW", np.array([1, 2, 3, 4, 5], dtype=np.int32)),  # broadcast
    ((2, 3), "HW", np.array([[1], [2]], dtype=np.int32)),  # broadcast
)
def test_full_layout(sh, layout, fill_value):
    arr, out_layout = run_with_layout(fn.full(fill_value, shape=sh, layout=layout))
    np.testing.assert_array_equal(arr, np.full(sh, fill_value))
    assert out_layout == layout


@params(
    ((2, 3, 4), "HWC", 42),
    ((3, 5), "HW", np.array([1, 2, 3, 4, 5], dtype=np.int32)),  # broadcast
    ((2, 3), "HW", np.array([[1], [2]], dtype=np.int32)),  # broadcast
)
def test_full_like_layout(sh, layout, fill_value):
    arr = types.Constant(np.ones(sh), layout=layout)
    result, out_layout = run_with_layout(fn.full_like(arr, fill_value))
    np.testing.assert_array_equal(result, np.full(sh, fill_value))
    assert out_layout == layout


@params(
    (fn.zeros, (2, 3, 4)),
    (fn.ones, (2, 3)),
    (fn.full, (5, 4, 3)),
)
def test_const_empty_layout(op, sh):
    op_to_run = op(42, shape=sh, layout="") if op == fn.full else op(shape=sh, layout="")
    arr, out_layout = run_with_layout(op_to_run)
    assert out_layout == ""


@params(
    (fn.zeros_like, (2, 3, 4)),
    (fn.ones_like, (2, 3)),
    (fn.full_like, (5, 4, 3)),
)
def test_const_like_empty_layout(op, sh):
    arr = types.Constant(np.ones(sh), layout="")
    result, out_layout = run_with_layout(op(arr, 42) if op == fn.full_like else op(arr))
    assert out_layout == ""


@params(
    (fn.zeros, (2, 3, 4), "HW", 2, 3),
    (fn.ones, (2, 3), "FHWC", 4, 2),
    (fn.full, (2, 3), "HWC", 3, 2),
)
def test_const_layout_mismatch(op, sh, layout, layout_ndim, shape_ndim):
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return op(42, shape=sh, layout=layout) if op == fn.full else op(shape=sh, layout=layout)

    p = pipe()
    p.build()
    assert_raises(
        RuntimeError,
        p.run,
        glob=f"Layout '{layout}' has {layout_ndim} dimensions but output shape has {shape_ndim}*",
    )
