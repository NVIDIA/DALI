# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itertools import product

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import eval_modes
from nose2.tools import params

# Non-zero values that are easy to lose in a conversion to bool:
# - integers with all-zero low bits would become False if truncated to a narrower type,
# - floats closer than 0.5 to zero would become False if rounded to an integer,
# - NaN compares unequal to zero, so it is True.
nonzero_values = {
    np.bool_: [True],
    np.uint8: [1, 128, 255],
    np.int8: [-128, -1, 1, 127],
    np.int32: [256, 65536, -(1 << 24), 1],
    np.int64: [1 << 32, -(1 << 40), 256, 1],
    np.float16: [0.001, -0.25, 0.4995, np.nan],
    np.float32: [1e-30, -0.25, 0.49, np.nan, -1e-5],
    np.float64: [1e-300, -0.3, 0.4999999, np.nan],
}

# Negative zero is still zero, so it is False.
zero_values = {
    np.bool_: [False],
    np.uint8: [0],
    np.int8: [0],
    np.int32: [0],
    np.int64: [0],
    np.float16: [0.0, -0.0],
    np.float32: [0.0, -0.0],
    np.float64: [0.0, -0.0],
}

# Masks of non-zero elements. Each mask has slices along both axes for which `all` and `any`
# give different results.
masks = [
    # rows: all set, none set, some set
    np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]], dtype=bool),
    # columns: all set, none set, some set
    np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=bool),
]

dtypes = list(nonzero_values.keys())


def make_data(mask, dtype):
    """Fills the non-zero positions of the mask with tricky non-zero values of given type."""
    nonzero = np.array(nonzero_values[dtype], dtype=dtype)
    zero = np.array(zero_values[dtype], dtype=dtype)
    data = np.empty(mask.shape, dtype=dtype)
    flat_mask = mask.ravel()
    flat_data = data.reshape(-1)
    n = np.count_nonzero(flat_mask)
    flat_data[flat_mask] = np.resize(nonzero, n)
    flat_data[~flat_mask] = np.resize(zero, flat_mask.size - n)
    return data


def check(result, ref):
    assert result.dtype == ndd.bool, f"Expected bool result, got {result.dtype}"
    actual = np.asarray(result.cpu())
    assert actual.shape == ref.shape, f"Shape mismatch: {actual.shape} vs {ref.shape}"
    assert np.array_equal(actual, ref), f"Result mismatch:\n{actual}\nvs\n{ref}"


def to_np_axis(axes):
    return tuple(axes) if isinstance(axes, list) else axes


functions = [(ndd.all, np.all), (ndd.any, np.any)]
axes_variants = [None, 0, 1, -1, [0, 1], []]


@eval_modes()
@params(*product(dtypes, ["cpu", "gpu"]))
def test_tensor(dtype, device_type):
    for mask in masks:
        data = make_data(mask, dtype)
        t = ndd.as_tensor(data, device=device_type)
        for (ndd_fn, np_fn), axes, keep_dims in product(functions, axes_variants, [False, True]):
            ref = np_fn(data, axis=to_np_axis(axes), keepdims=keep_dims)
            if axes is None and not keep_dims:
                check(ndd_fn(t), ref)
            else:
                check(ndd_fn(t, axes=axes, keep_dims=keep_dims), ref)


@eval_modes()
@params(*product(dtypes, ["cpu", "gpu"]))
def test_batch(dtype, device_type):
    samples = [make_data(mask, dtype) for mask in masks]
    samples.append(np.zeros((0, 3), dtype=dtype))  # empty sample
    b = ndd.as_batch(samples, device=device_type)
    for (ndd_fn, np_fn), axes in product(functions, [None, 0, 1]):
        result = ndd_fn(b) if axes is None else ndd_fn(b, axes=axes)
        assert isinstance(result, ndd.Batch)
        assert result.batch_size == len(samples)
        for i, sample in enumerate(samples):
            check(result.tensors[i], np_fn(sample, axis=axes))


@eval_modes()
@params(("cpu",), ("gpu",))
def test_empty(device_type):
    t = ndd.as_tensor(np.zeros((0, 5), dtype=np.float32), device=device_type)
    check(ndd.all(t), np.array(True))
    check(ndd.any(t), np.array(False))
    check(ndd.all(t, axes=0), np.ones(5, dtype=bool))
    check(ndd.any(t, axes=0), np.zeros(5, dtype=bool))
    check(ndd.all(t, axes=1), np.zeros(0, dtype=bool))


@eval_modes()
@params(*product([np.bool_, np.float32], ["cpu", "gpu"]))
def test_large(dtype, device_type):
    # Large enough to be reduced in multiple stages on the GPU
    shape = (3, 300001)
    for pos in [0, shape[1] // 2, shape[1] - 1]:
        sparse = np.zeros(shape, dtype=dtype)
        sparse[1, pos] = 0.25 if dtype == np.float32 else True
        dense = np.full(shape, 0.25 if dtype == np.float32 else True, dtype=dtype)
        dense[1, pos] = 0
        for data in [sparse, dense]:
            t = ndd.as_tensor(data, device=device_type)
            for (ndd_fn, np_fn), axes in product(functions, [None, 1]):
                result = ndd_fn(t) if axes is None else ndd_fn(t, axes=axes)
                check(result, np_fn(data, axis=axes))


@eval_modes()
@params(("cpu",), ("gpu",))
def test_axis_names(device_type):
    data = make_data(masks[0], np.float32)
    t = ndd.as_tensor(data, device=device_type, layout="HW")
    for ndd_fn, np_fn in functions:
        check(ndd_fn(t, axis_names="W"), np_fn(data, axis=1))
        check(ndd_fn(t, axis_names="H", keep_dims=True), np_fn(data, axis=0, keepdims=True))
        check(ndd_fn(t, axis_names="HW"), np_fn(data))


@eval_modes()
def test_non_dali_input():
    data = make_data(masks[0], np.float32)
    for ndd_fn, np_fn in functions:
        check(ndd_fn(data, axes=1), np_fn(data, axis=1))
        check(ndd_fn(data.tolist(), axes=1), np_fn(data, axis=1))
        check(ndd_fn(0.25), np_fn(0.25))
        check(ndd_fn(-0.0), np_fn(-0.0))
