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

def exp(value):
    """Fills the output with exponential of value.
    :rtype: TensorList of exp(value). If value is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("exp", value)

def log(value):
    """Fills the output with logarithm (base-10) of value.
    :rtype: TensorList of log(value). If value is an integer, the result will be float, otherwise the type is preserved.
    """
    return _arithm_op("log", value)

def min(left, right):
    """Fills the output with minima of corresponding values in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("min", left, right)

def max(left, right):
    """Fills the output with maxima of corresponding values in ``left`` and ``right``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("max", left, right)

def clamp(value, lo, hi):
    """Produces a tensor of values from ``value`` clamped to the range ``[lo, hi]``.

    :rtype: TensorList of the type that is calculated based on the type promotion rules.
    """
    return _arithm_op("clamp", value, lo, hi)
