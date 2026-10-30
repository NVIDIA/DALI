# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali import fn, ops, Pipeline
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
        "sub.sub.DeprecationWarningOp",
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
        "sub.sub.DeprecationWarningOp",
        "",
    ),
]


@params(*module_variants)
def test_warnings(op, name, replacement, message):

    glob = f"WARNING: `{op.__module__}.{name}` is now deprecated."
    if replacement:
        # The replacement is reported through whichever API namespace (`fn` or `ops`) the
        # deprecated operator was actually called through.
        api = "ops" if op.__module__.startswith("nvidia.dali.ops") else "fn"
        glob += f" Use `nvidia.dali.{api}.{replacement}` instead."
    if message:
        glob += f"\n{message}"
    with assert_warns(DeprecationWarning, glob=glob):
        op()


alias_variants = [
    (fn.mxnet_reader, "nvidia.dali.fn.mxnet_reader", "readers.mxnet"),
    (ops.MXNetReader, "nvidia.dali.ops.MXNetReader", "readers.mxnet"),
    (fn.numpy_reader, "nvidia.dali.fn.numpy_reader", "readers.numpy"),
    (fn.experimental.tensor_resize, "nvidia.dali.fn.experimental.tensor_resize", "tensor_resize"),
    (
        ops.experimental.TensorResize,
        "nvidia.dali.ops.experimental.TensorResize",
        "tensor_resize",
    ),
]


def _call_op(op, *inputs, **kwargs):
    if isinstance(op, type):
        return op(**kwargs)(*inputs)
    else:
        return op(*inputs, **kwargs)


@params(*alias_variants)
def test_alias_warnings(op, full_name, replacement):
    glob = f"WARNING: `{full_name}` is now deprecated. Use `nvidia.dali.fn.{replacement}` instead.*"
    with Pipeline(batch_size=1, num_threads=1, device_id=None):
        if "reader" in full_name.lower():
            inputs = []
            kwargs = {"path": "dummy", "index_path": "dummy"} if "mxnet" in full_name else {}
        else:
            inputs = [fn.external_source(name="input")]
            kwargs = {"sizes": [1]}
        with assert_warns(DeprecationWarning, glob=glob):
            _call_op(op, *inputs, **kwargs)
