# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali._utils import dali_trace as _dali_trace


def _arithm_op(*args, **kwargs):
    import nvidia.dali.ops

    if _dali_trace.is_tracing_enabled():
        definition_frame_end = _dali_trace.get_stack_depth() - 2
    else:
        definition_frame_end = None

    # Fully circular imports don't work. We need to import _arithm_op late and
    # replace this trampoline function.
    setattr(sys.modules[__name__], "_arithm_op", nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs, definition_frame_end=definition_frame_end)


def sqrt(input) -> _DataNode:
    """Computes square root of values in `input`.

    :rtype: TensorList of sqrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sqrt", input)


def rsqrt(input) -> _DataNode:
    """Computes reciprocal of the square root of values in `input`.

    :rtype: TensorList of rsqrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("rsqrt", input)


def cbrt(input) -> _DataNode:
    """Computes cube root of values in `input`.

    :rtype: TensorList of cbrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cbrt", input)


def exp(input) -> _DataNode:
    """Computes exponential of values in `input`.

    :rtype: TensorList of exp(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("exp", input)


def log(input) -> _DataNode:
    """Computes natural logarithm (base-e) of values in `input`.

    :rtype: TensorList of log(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log", input)


def log2(input) -> _DataNode:
    """Computes logarithm (base-2) of values in `input`.

    :rtype: TensorList of log2(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log2", input)


def log10(input) -> _DataNode:
    """Computes logarithm (base-10) of values in `input`.

    :rtype: TensorList of log10(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log10", input)


def abs(input) -> _DataNode:
    """Computes absolute value of values in `input`.

    :rtype: TensorList of abs(input). The type is preserved.
    """
    return _arithm_op("abs", input)


def fabs(input) -> _DataNode:
    """Computes float absolute value of values in `input`.

    :rtype: TensorList of fabs(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("fabs", input)


def floor(input) -> _DataNode:
    """Computes floor of values in `input`.

    :rtype: TensorList of floor(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("floor", input)


def ceil(input) -> _DataNode:
    """Computes ceil of values in `input`.

    :rtype: TensorList of ceil(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("ceil", input)


def sin(input) -> _DataNode:
    """Computes sine of values in `input`.

    :rtype: TensorList of sin(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sin", input)


def cos(input) -> _DataNode:
    """Computes cosine of values in `input`.

    :rtype: TensorList of cos(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cos", input)


def tan(input) -> _DataNode:
    """Computes tangent of values in `input`.

    :rtype: TensorList of tan(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("tan", input)


def asin(input) -> _DataNode:
    """Computes arcus sine of values in `input`.

    :rtype: TensorList of asin(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("asin", input)


def acos(input) -> _DataNode:
    """Computes arcus cosine of values in `input`.

    :rtype: TensorList of acos(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("acos", input)


def atan(input) -> _DataNode:
    """Computes arcus tangent of values in `input`.

    :rtype: TensorList of atan(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atan", input)


def sinh(input) -> _DataNode:
    """Computes hyperbolic sine of values in `input`.

    :rtype: TensorList of sinh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sinh", input)


def cosh(input) -> _DataNode:
    """Computes hyperbolic cosine of values in `input`.

    :rtype: TensorList of cosh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cosh", input)


def tanh(input) -> _DataNode:
    """Computes hyperbolic tangent of values in `input`.

    :rtype: TensorList of tanh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("tanh", input)


def asinh(input) -> _DataNode:
    """Computes inverse hyperbolic sine of values in `input`.

    :rtype: TensorList of asinh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("asinh", input)


def acosh(input) -> _DataNode:
    """Computes inverse hyperbolic cosine of values in `input`.

    :rtype: TensorList of acosh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("acosh", input)


def atanh(input) -> _DataNode:
    """Computes inverse hyperbolic tangent of values in `input`.

    :rtype: TensorList of atanh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atanh", input)


def min(left, right) -> _DataNode:
    """Computes minima of corresponding values in `left` and `right`.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("min", left, right)


def max(left, right) -> _DataNode:
    """Computes maxima of corresponding values in `left` and `right`.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("max", left, right)


def pow(base, exponent) -> _DataNode:
    """Computes base to the power of exponents, that is base ** exponent.

    :rtype: TensorList of pow(base, exponent). Type is calculated based on the type promotion rules.
    """
    return _arithm_op("pow", base, exponent)


def fpow(base, exponent) -> _DataNode:
    """Computes base to the power of exponents as floating point numbers.

    :rtype: TensorList of pow(base, exponent). If all inputs are integers, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("fpow", base, exponent)


def atan2(x, y) -> _DataNode:
    """Computes arcus tangent of corresponding values in  x / y.

    :rtype: TensorList of atan2(x, y). If all inputs are integers, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atan2", x, y)


def clamp(value, lo, hi) -> _DataNode:
    """Produces a tensor of values from `value` clamped to the range ``[lo, hi]``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("clamp", value, lo, hi)
