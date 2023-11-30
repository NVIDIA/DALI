# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Conditional expressions (e.g. the ternary if statement)."""

from nvidia.dali._autograph.utils import hooks


def if_exp(cond, if_true, if_false, expr_repr):
    if hooks._DISPATCH.detect_overload_if_exp(cond):
        return hooks._DISPATCH.if_exp(cond, if_true, if_false, expr_repr)
    else:
        return _py_if_exp(cond, if_true, if_false)


def _py_if_exp(cond, if_true, if_false):
    return if_true() if cond else if_false()
