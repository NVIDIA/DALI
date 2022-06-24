# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._utils.eager_utils import _arithm_op

# Eager version of math operators. Implements same operators as `nvidia.dali.math`,
# but working directly on TensorLists.


def sqrt(input):
    return _arithm_op("sqrt", input)


def rsqrt(input):
    return _arithm_op("rsqrt", input)


def cbrt(input):
    return _arithm_op("cbrt", input)


def exp(input):
    return _arithm_op("exp", input)


def log(input):
    return _arithm_op("log", input)


def log2(input):
    return _arithm_op("log2", input)


def log10(input):
    return _arithm_op("log10", input)


def abs(input):
    return _arithm_op("abs", input)


def fabs(input):
    return _arithm_op("fabs", input)


def floor(input):
    return _arithm_op("floor", input)


def ceil(input):
    return _arithm_op("ceil", input)


def sin(input):
    return _arithm_op("sin", input)


def cos(input):
    return _arithm_op("cos", input)


def tan(input):
    return _arithm_op("tan", input)


def asin(input):
    return _arithm_op("asin", input)


def acos(input):
    return _arithm_op("acos", input)


def atan(input):
    return _arithm_op("atan", input)


def sinh(input):
    return _arithm_op("sinh", input)


def cosh(input):
    return _arithm_op("cosh", input)


def tanh(input):
    return _arithm_op("tanh", input)


def asinh(input):
    return _arithm_op("asinh", input)


def acosh(input):
    return _arithm_op("acosh", input)


def atanh(input):
    return _arithm_op("atanh", input)


def min(left, right):
    return _arithm_op("min", left, right)


def max(left, right):
    return _arithm_op("max", left, right)


def pow(base, exponent):
    return _arithm_op("pow", base, exponent)


def fpow(base, exponent):
    return _arithm_op("fpow", base, exponent)


def atan2(x, y):
    return _arithm_op("atan2", x, y)


def clamp(value, lo, hi):
    return _arithm_op("clamp", value, lo, hi)
