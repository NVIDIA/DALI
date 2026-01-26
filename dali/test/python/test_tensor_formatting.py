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

import numpy as np
from itertools import product
from nvidia.dali.backend_impl import TensorCPU, TensorListCPU
from nose2.tools import params

# Import Dynamic API
import nvidia.dali.experimental.dynamic as ndd


# Test case definitions for parameterized tests
TENSOR_TEST_CASES = [
    {
        "name": "simple_tensor",
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
        "layout": "HW",
        "expected_shape": "(3, 4)",
    },
    {
        "name": "large_tensor",
        "array": np.arange(1250, dtype=np.int32).reshape(25, 50),
        "layout": None,
        "expected_shape": "(25, 50)",
    },
    {
        "name": "1d_tensor",
        "array": np.array([5.0], dtype=np.float32),
        "layout": None,
        "expected_shape": "(1,)",
    },
    {
        "name": "scalar_tensor",
        "array": np.array(5.0, dtype=np.float32),
        "layout": None,
        "expected_shape": "()",
    },
    {
        "name": "empty_tensor",
        "array": np.empty(0, dtype=np.float32),
        "layout": None,
        "expected_shape": "(0,)",
    },
]

BATCH_TEST_CASES = [
    {
        "name": "dense_batch_small",
        "arrays": [np.arange(6, dtype=np.float32).reshape(2, 3) + i * 6 for i in range(4)],
        "layout": "HW",
        "num_samples": 4,
        "shapes": ["(2, 3)"] * 4,
        "should_summarize": False,
    },
    {
        "name": "dense_batch_large",
        "arrays": [np.ones((10, 10), dtype=np.float32) for _ in range(12)],
        "layout": None,
        "num_samples": 12,
        "shapes": ["(10, 10)"] * 12,
        "should_summarize": True,
    },
    {
        "name": "dense_batch_large_samples",
        "arrays": [np.ones((25, 50), dtype=np.int32) for _ in range(6)],
        "layout": None,
        "num_samples": 6,
        "shapes": ["(25, 50)"] * 6,
        "should_summarize": True,
    },
    {
        "name": "non_dense_batch_small",
        "arrays": [
            np.ones((2, 3), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ],
        "layout": None,
        "num_samples": 3,
        "shapes": ["(2, 3)", "(3, 4)", "(1, 2)"],
        "should_summarize": False,
    },
    {
        "name": "empty_batch",
        "arrays": [],
        "layout": None,
        "num_samples": 0,
        "shapes": [],
        "should_summarize": False,
    },
]


DEVICES = ["cpu", "gpu"]


def _format_values(arr, count=2):
    """Format array values as strings matching the output format."""
    return [
        np.format_float_positional(v, trim="-") if arr.dtype.kind == "f" else str(int(v))
        for v in arr.flat[:count]
    ]


@params(*product(TENSOR_TEST_CASES, DEVICES))
def test_tensor_formatting_pipeline(test_case, device):
    """Test tensor formatting for Pipeline API."""

    arr = test_case["array"].copy()
    layout = test_case["layout"]
    expected_shape = test_case["expected_shape"]

    t = TensorCPU(arr, layout) if layout else TensorCPU(arr)
    if device == "cpu":
        type_name = "TensorCPU"
    else:
        t = t._as_gpu()
        type_name = "TensorGPU"

    for method_name, method in [("str", str), ("repr", repr)]:
        result = method(t)
        assert type_name in result, f'expected "{type_name}" in "{result}"'
        assert "dtype=" in result, f'expected "dtype=" in "{result}"'
        assert f'device="{device}"' in result, f'expected "device=\\"{device}\\"" in "{result}"'
        assert (
            f"shape={expected_shape}" in result
        ), f'expected "shape={expected_shape}" in "{result}"'
        if layout:
            assert f'layout="{layout}"' in result, f'expected "layout=\\"{layout}\\"" in "{result}"'

        if method_name == "str" and arr.size > 0 and arr.ndim > 0:
            assert "[" in result, f'expected "[" in "{result}"'
        elif method_name == "repr":
            assert (
                "[" not in result or result.count("[") <= 1
            ), f'expected no data arrays in "{result}"'


@params(*product(TENSOR_TEST_CASES, DEVICES))
def test_tensor_formatting_dynamic(test_case, device):
    """Test tensor formatting for Dynamic API."""
    arr = test_case["array"].copy()
    layout = test_case["layout"]
    expected_shape = test_case["expected_shape"]

    t = ndd.tensor(arr, layout=layout, device=device) if layout else ndd.tensor(arr, device=device)

    for method_name, method in [("str", str), ("repr", repr)]:
        result = method(t)
        assert "Tensor" in result, f'expected "Tensor" in "{result}"'
        assert "dtype=" in result, f'expected "dtype=" in "{result}"'
        assert f'device="{device}"' in result, f'expected "device=\\"{device}\\"" in "{result}"'
        assert (
            f"shape={expected_shape}" in result
        ), f'expected "shape={expected_shape}" in "{result}"'
        if layout:
            assert f'layout="{layout}"' in result, f'expected "layout=\\"{layout}\\"" in "{result}"'
        if arr.size > 0 and arr.ndim > 0:
            assert "[" in result, f'expected "[" in "{result}"'


@params(*product(BATCH_TEST_CASES, DEVICES))
def test_batch_formatting_pipeline(test_case, device):
    """Test batch formatting for Pipeline API."""

    layout = test_case["layout"]
    num_samples = test_case["num_samples"]
    shapes = test_case["shapes"]
    should_summarize = test_case["should_summarize"]
    arrays = test_case["arrays"]

    if num_samples == 0:
        b = TensorListCPU(np.empty(0))
    else:
        tensors = [TensorCPU(arr, layout) for arr in arrays]
        b = TensorListCPU(tensors)

    if device == "cpu":
        type_name = "TensorListCPU"
    else:
        b = b._as_gpu()
        type_name = "TensorListGPU"

    for method_name, method in [("str", str), ("repr", repr)]:
        result = method(b)
        assert type_name in result, f'expected "{type_name}" in "{result}"'
        assert "dtype=" in result, f'expected "dtype=" in "{result}"'
        assert f'device="{device}"' in result, f'expected "device=\\"{device}\\"" in "{result}"'
        assert (
            f"num_samples={num_samples}" in result
        ), f'expected "num_samples={num_samples}" in "{result}"'
        if layout:
            assert f'layout="{layout}"' in result, f'expected "layout=\\"{layout}\\"" in "{result}"'

        if method_name == "str":
            if num_samples > 0:
                assert result.count("[") > 1, f'expected multiple "[" in "{result}"'
                assert (
                    shapes[0] in result and shapes[-1] in result
                ), f'expected "{shapes[0]}" and "{shapes[-1]}" in "{result}"'
                expected_vals = _format_values(arrays[0]) + _format_values(arrays[-1])
                assert all(
                    v in result for v in expected_vals
                ), f'expected values {expected_vals} in "{result}"'
            if should_summarize:
                assert "..." in result, f'expected "..." in "{result}"'


@params(*product(BATCH_TEST_CASES, DEVICES))
def test_batch_formatting_dynamic(test_case, device):
    """Test batch formatting for Dynamic API."""
    layout = test_case["layout"]
    num_samples = test_case["num_samples"]
    shapes = test_case["shapes"]
    should_summarize = test_case["should_summarize"]
    arrays = test_case["arrays"]

    if num_samples == 0:
        import nvidia.dali.types as types

        b = ndd.batch([], dtype=types.FLOAT, device=device)
    else:
        b = ndd.batch(arrays, layout=layout, device=device)

    for method_name, method in [("str", str), ("repr", repr)]:
        result = method(b)
        assert "Batch" in result, f'expected "Batch" in "{result}"'
        assert "dtype=" in result, f'expected "dtype=" in "{result}"'
        assert f'device="{device}"' in result, f'expected "device=\\"{device}\\"" in "{result}"'
        assert (
            f"num_samples={num_samples}" in result
        ), f'expected "num_samples={num_samples}" in "{result}"'
        if layout:
            assert f'layout="{layout}"' in result, f'expected "layout=\\"{layout}\\"" in "{result}"'

        if num_samples > 0:
            assert result.count("[") > 1, f'expected multiple "[" in "{result}"'
            assert (
                shapes[0] in result and shapes[-1] in result
            ), f'expected "{shapes[0]}" and "{shapes[-1]}" in "{result}"'
            expected_vals = _format_values(arrays[0]) + _format_values(arrays[-1])
            assert all(
                v in result for v in expected_vals
            ), f'expected values {expected_vals} in "{result}"'
        if should_summarize:
            assert "..." in result, f'expected "..." in "{result}"'
