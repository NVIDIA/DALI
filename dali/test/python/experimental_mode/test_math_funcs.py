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

import nvidia.dali.experimental.dynamic as ndd
import numpy as np
from nose2.tools import params
from itertools import product
from nose_utils import assert_raises

arbitrary = "arbitrary"
nonzero = "nonzero"
nonnegative = "nonnegative"
positive = "positive"
abs1to1 = "abs1to1"
over1 = "over1"

unary_functions = [
    (ndd.math.sqrt, np.sqrt, nonnegative),
    (ndd.math.rsqrt, lambda x: 1.0 / np.sqrt(x), positive),
    (ndd.math.cbrt, np.cbrt, arbitrary),
    (ndd.math.exp, np.exp, arbitrary),
    (ndd.math.log, np.log, positive),
    (ndd.math.log2, np.log2, positive),
    (ndd.math.log10, np.log10, positive),
    (ndd.math.abs, np.abs, arbitrary),
    (ndd.math.fabs, np.fabs, arbitrary),
    (ndd.math.floor, np.floor, arbitrary),
    (ndd.math.ceil, np.ceil, arbitrary),
    (ndd.math.sin, np.sin, arbitrary),
    (ndd.math.cos, np.cos, arbitrary),
    (ndd.math.tan, np.tan, arbitrary),
    (ndd.math.asin, np.arcsin, abs1to1),
    (ndd.math.acos, np.arccos, abs1to1),
    (ndd.math.atan, np.arctan, arbitrary),
    (ndd.math.sinh, np.sinh, arbitrary),
    (ndd.math.cosh, np.cosh, arbitrary),
    (ndd.math.tanh, np.tanh, arbitrary),
    (ndd.math.asinh, np.arcsinh, arbitrary),
    (ndd.math.acosh, np.arccosh, over1),
    (ndd.math.atanh, np.arctanh, abs1to1),
]

binary_functions = [
    (ndd.math.pow, lambda x, y: np.power(x, y), arbitrary, nonzero),
    (ndd.math.fpow, np.power, arbitrary, nonzero),
    (ndd.math.atan2, np.arctan2, arbitrary, arbitrary),
]


def get_operand1(domain):
    if domain == arbitrary:
        return np.array([[-10, -1, 0, 1, 10], [0.1, 0.25, 0.5, 2.5, 12]], dtype=np.float32)
    elif domain == nonzero:
        return np.array([[-10, -1, 0.1, 1, 10], [0.1, 0.25, 0.5, 2.5, 12]], dtype=np.float32)
    elif domain == nonnegative:
        return np.array([[0, 0.1, 0.25], [0.5, 2.5, 12]], dtype=np.float32)
    elif domain == positive:
        return np.array([[0.1, 0.25, 0.5], [1.25, 2.5, 12]], dtype=np.float32)
    elif domain == abs1to1:
        return np.array([[-0.99, -0.5, -0.25], [0.25, 0.5, 0.99]], dtype=np.float32)
    elif domain == over1:
        return np.array([[1.1, 2, 3], [4, 5, 6]], dtype=np.float32)


def get_operand2(domain):
    if domain == arbitrary:
        return np.array([[-9, -1, 0, 1, 11], [0.1, 0.3, 0.55, 2.8, 15]], dtype=np.float32)
    elif domain == nonzero:
        return np.array([[-11, -1, 0.2, 1, 9], [0.1, 0.5, 1.5, 2.5, 13]], dtype=np.float32)
    elif domain == nonnegative:
        return np.array([[0, 0.2, 0.3], [0.5, 3, 10]], dtype=np.float32)
    elif domain == positive:
        return np.array([[0.1, 0.22, 0.5], [1.5, 3.5, 10]], dtype=np.float32)
    elif domain == abs1to1:
        return np.array([[-0.98, -0.35, -0.25], [0.7, 0.5, 0.99]], dtype=np.float32)
    elif domain == over1:
        return np.array([[1.1, 2, 2.5], [5, 7, 10]], dtype=np.float32)


@params(*product(unary_functions, ["cpu", "gpu"]))
def test_unary_functions(functions, device_type):
    ndd_func, np_func, domain = functions
    data = get_operand1(domain)
    t = ndd.Tensor(data, device=device_type)
    a = np.array(ndd_func(t).cpu())
    ref = np_func(data)
    assert np.allclose(a, ref, atol=1e-6)


@params(*product(binary_functions, ["cpu", "gpu"]))
def test_binary_functions(functions, device_type):
    ndd_func, np_func, domain1, domain2 = functions
    data1 = get_operand1(domain1)
    data2 = get_operand2(domain2)
    t1 = ndd.Tensor(data1, device=device_type)
    t2 = ndd.Tensor(data2, device=device_type)
    a = np.array(ndd_func(t1, t2).cpu())
    ref = np_func(data1, data2)
    assert np.allclose(a, ref, atol=1e-6)


@params(*product(binary_functions, ["cpu", "gpu"]))
def test_binary_functions_batch(functions, device_type):
    ndd_func, np_func, domain1, domain2 = functions
    data1 = get_operand1(domain1)
    data2 = get_operand2(domain2)
    t1 = ndd.as_batch(data1, device=device_type)
    t2 = ndd.as_batch(data2, device=device_type)
    a = ndd_func(t1, t2)
    ref = np_func(data1, data2)
    assert np.allclose(a.tensors[0].cpu(), ref[0], atol=1e-6)
    assert np.allclose(a.tensors[1].cpu(), ref[1], atol=1e-6)


@params(("cpu",), ("gpu",))
def test_clamp(device_type):
    data = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.float32)
    t = ndd.Tensor(data, device=device_type)
    a = np.array(ndd.math.clamp(t, 1, 2).cpu())
    ref = np.clip(data, 1, 2)
    assert np.allclose(a, ref, atol=1e-6)


def test_mixed_operand_devices_error():
    data1 = np.array([1, 2, 3], dtype=np.float32)
    data2 = np.array([4, 5, 6], dtype=np.float32)
    t1 = ndd.Tensor(data1, device="cpu")
    t2 = ndd.Tensor(data2, device="gpu")
    with assert_raises(ValueError, glob="Cannot mix GPU and CPU inputs"):
        _ = ndd.math.pow(t1, t2).cpu()
