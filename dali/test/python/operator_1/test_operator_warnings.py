# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali import fn, ops
from nose2.tools import params
from test_utils import load_test_operator_plugin
from nose_utils import assert_warns

load_test_operator_plugin()


module_variants = [
    (
        fn.deprecation_warning_op,
        "deprecation_warning_op",
        "",
        "Additional message",
    ),
    (ops.DeprecationWarningOp(), "DeprecationWarningOp", "", "Additional message"),
    (
        fn.sub.deprecation_warning_op,
        "deprecation_warning_op",
        "sub.sub.deprecation_warning_op",
        "Another message",
    ),
    (
        ops.sub.DeprecationWarningOp(),
        "DeprecationWarningOp",
        "sub.sub.deprecation_warning_op",
        "Another message",
    ),
    (
        fn.sub.sub.deprecation_warning_op,
        "deprecation_warning_op",
        "sub.sub.deprecation_warning_op",
        "",
    ),
    (
        ops.sub.sub.DeprecationWarningOp(),
        "DeprecationWarningOp",
        "sub.sub.deprecation_warning_op",
        "",
    ),
]


@params(*module_variants)
def test_warnings(op, name, replacement, message):

    glob = f"WARNING: `{op.__module__}.{name}` is now deprecated."
    if replacement:
        glob += f" Use `nvidia.dali.fn.{replacement}` instead."
    if message:
        glob += f"\n{message}"
    with assert_warns(DeprecationWarning, glob=glob):
        op()
