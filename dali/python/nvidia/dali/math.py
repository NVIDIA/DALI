# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


def _arithm_op(*args, **kwargs):
    import nvidia.dali.ops
    # Fully circular imports don't work. We need to import _arithm_op late and
    # replace this trampoline function.
    setattr(sys.modules[__name__], "_arithm_op", nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs)


def sqrt(input):
    """Computes square root of values in ``input``.

    :rtype: TensorList of sqrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sqrt", input)


def rsqrt(input):
    """Computes reciprocal of the square root of values in ``input``.

    :rtype: TensorList of rsqrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("rsqrt", input)


def cbrt(input):
    """Computes cube root of values in ``input``.

    :rtype: TensorList of cbrt(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cbrt", input)


def exp(input):
    """Computes exponential of values in ``input``.

    :rtype: TensorList of exp(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("exp", input)


def log(input):
    """Computes natural logarithm (base-e) of values in ``input``.

    :rtype: TensorList of log(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log", input)


def log2(input):
    """Computes logarithm (base-2) of values in ``input``.

    :rtype: TensorList of log2(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log2", input)


def log10(input):
    """Computes logarithm (base-10) of values in ``input``.

    :rtype: TensorList of log10(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("log10", input)


def abs(input):
    """Computes absolute value of values in ``input``.

    :rtype: TensorList of abs(input). The type is preserved.
    """
    return _arithm_op("abs", input)


def fabs(input):
    """Computes float absolute value of values in ``input``.

    :rtype: TensorList of fabs(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("fabs", input)


def floor(input):
    """Computes floor of values in ``input``.

    :rtype: TensorList of floor(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("floor", input)


def ceil(input):
    """Computes ceil of values in ``input``.

    :rtype: TensorList of ceil(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("ceil", input)


def sin(input):
    """Computes sine of values in ``input``.

    :rtype: TensorList of sin(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sin", input)


def cos(input):
    """Computes cosine of values in ``input``.

    :rtype: TensorList of cos(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cos", input)


def tan(input):
    """Computes tangent of values in ``input``.

    :rtype: TensorList of tan(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("tan", input)


def asin(input):
    """Computes arcus sine of values in ``input``.

    :rtype: TensorList of asin(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("asin", input)


def acos(input):
    """Computes arcus cosine of values in ``input``.

    :rtype: TensorList of acos(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("acos", input)


def atan(input):
    """Computes arcus tangent of values in ``input``.

    :rtype: TensorList of atan(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atan", input)


def sinh(input):
    """Computes hyperbolic sine of values in ``input``.

    :rtype: TensorList of sinh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("sinh", input)


def cosh(input):
    """Computes hyperbolic cosine of values in ``input``.

    :rtype: TensorList of cosh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("cosh", input)


def tanh(input):
    """Computes hyperbolic tangent of values in ``input``.

    :rtype: TensorList of tanh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("tanh", input)


def asinh(input):
    """Computes inverse hyperbolic sine of values in ``input``.

    :rtype: TensorList of asinh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("asinh", input)


def acosh(input):
    """Computes inverse hyperbolic cosine of values in ``input``.

    :rtype: TensorList of acosh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("acosh", input)


def atanh(input):
    """Computes inverse hyperbolic tangent of values in ``input``.

    :rtype: TensorList of atanh(input). If input is an integer, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atanh", input)


def min(left, right):
    """Computes minima of corresponding values in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("min", left, right)


def max(left, right):
    """Computes maxima of corresponding values in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("max", left, right)


def pow(base, exponent):
    """Computes base to the power of exponents, that is base ** exponent.

    :rtype: TensorList of pow(base, exponent). Type is calculated based on the type promotion rules.
    """
    return _arithm_op("pow", base, exponent)


def fpow(base, exponent):
    """Computes base to the power of exponents as floating point numbers.

    :rtype: TensorList of pow(base, exponent). If all inputs are integers, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("fpow", base, exponent)


def atan2(x, y):
    """Computes arcus tangent of corresponding values in  x / y.

    :rtype: TensorList of atan2(x, y). If all inputs are integers, the result will be float,
            otherwise the type is preserved.
    """
    return _arithm_op("atan2", x, y)


def clamp(value, lo, hi):
    """Produces a tensor of values from ``value`` clamped to the range ``[lo, hi]``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("clamp", value, lo, hi)
