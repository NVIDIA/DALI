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
    """Fills the output with square root of the input.
    :rtype: TensorList of sqrt(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("sqrt", input)

def cbrt(input):
    """Fills the output with cube root of the input.
    :rtype: TensorList of cbrt(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("cbrt", input)

def exp(input):
    """Fills the output with exponential of the input.
    :rtype: TensorList of exp(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("exp", input)

def log(input):
    """Fills the output with natural logarithm (base-e) of the input.
    :rtype: TensorList of log(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("log", input)

def log2(input):
    """Fills the output with logarithm (base-2) of the input.
    :rtype: TensorList of log2(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("log2", input)

def log10(input):
    """Fills the output with logarithm (base-10) of the input.
    :rtype: TensorList of log10(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("log10", input)

def abs(input):
    """Fills the output with absolute value of the input.
    :rtype: TensorList of abs(input). The type is preserved.
    """
    return _arithm_op("abs", input)

def fabs(input):
    """Fills the output with float absolute value of the input.
    :rtype: TensorList of fabs(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("fabs", input)

def floor(input):
    """Fills the output with floor of the input.
    :rtype: TensorList of floor(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("floor", input)

def ceil(input):
    """Fills the output with ceil of the input.
    :rtype: TensorList of ceil(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("ceil", input)

def sin(input):
    """Fills the output with sine of the input.
    :rtype: TensorList of sin(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("sin", input)

def cos(input):
    """Fills the output with cosine of the input.
    :rtype: TensorList of cos(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("cos", input)

def tan(input):
    """Fills the output with tangent of the input.
    :rtype: TensorList of tan(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("tan", input)

def asin(input):
    """Fills the output with arcus sine of the input.
    :rtype: TensorList of asin(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("asin", input)

def acos(input):
    """Fills the output with arcus cosine of the input.
    :rtype: TensorList of acos(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("acos", input)

def atan(input):
    """Fills the output with arcus tangent of the input.
    :rtype: TensorList of atan(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("atan", input)

def sinh(input):
    """Fills the output with hyperbolic sine of the input.
    :rtype: TensorList of sinh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("sinh", input)

def cosh(input):
    """Fills the output with hyperbolic cosine of the input.
    :rtype: TensorList of cosh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("cosh", input)

def tanh(input):
    """Fills the output with hyperbolic tangent of the input.
    :rtype: TensorList of tanh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("tanh", input)

def asinh(input):
    """Fills the output with inverse hyperbolic sine of the input.
    :rtype: TensorList of asinh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("asinh", input)

def acosh(input):
    """Fills the output with inverse hyperbolic cosine of the input.
    :rtype: TensorList of acosh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("acosh", input)

def atanh(input):
    """Fills the output with inverse hyperbolic tangent of the input.
    :rtype: TensorList of atanh(input). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("atanh", input)

def min(left, right):
    """Fills the output with minima of corresponding inputs in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("min", left, right)

def max(left, right):
    """Fills the output with maxima of corresponding inputs in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("max", left, right)

def pow(base, exponent):
    """Fills the output with base to the power of exponents, that is base ** exponent.

    :rtype: TensorList of pow(base, exponent). If input is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("pow", base, exponent)


def clamp(value, lo, hi):
    """Produces a tensor of values from ``value`` clamped to the range ``[lo, hi]``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("clamp", value, lo, hi)
