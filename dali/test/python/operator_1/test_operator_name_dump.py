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

import numpy as np

from nvidia.dali import pipeline_def, fn, ops
from nose2.tools import params, cartesian_params
from test_utils import load_test_operator_plugin

load_test_operator_plugin()


def extract_str_from_tl(out):
    """Extract string from the test operator that returns it as u8 tensor."""
    # Extract data from first sample
    arr = np.array(out[0])
    if arr.shape[0] == 0:
        return ""
    # View it as list of characters and decode to string
    return arr.view(f"S{arr.shape[0]}")[0].decode("utf-8")


api_variants = [
    ("fn", "", fn.name_dump),
    ("fn", "sub", fn.sub.name_dump),
    ("fn", "sub.sub", fn.sub.sub.name_dump),
    ("ops", "", ops.NameDump()),
    ("ops", "sub", ops.sub.NameDump()),
    ("ops", "sub.sub", ops.sub.sub.NameDump()),
]


@params(*api_variants)
def test_api_name(api, module, op):
    del module

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="api")

    p = pipe()
    p.build()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    assert out_str == api, f"Expected {api}, got {out_str}"


module_kinds = ["Module", "ApiModule", "LibApiModule"]
op_name_kinds = ["OpOnly"] + module_kinds


def baseline_module_path(api, module, kind):
    if kind == "Module":
        return module
    elif kind == "ApiModule":
        dot = "." if module else ""
        return f"{api}{dot}{module}"
    elif kind == "LibApiModule":
        dot = "." if module else ""
        return f"nvidia.dali.{api}{dot}{module}"
    raise ValueError(f"Wrong kind: {kind}")


def baseline_display_name(api, module, kind):
    op_name = "name_dump" if api == "fn" else "NameDump"
    if kind == "OpOnly":
        return op_name
    else:
        module_str = baseline_module_path(api, module, kind)
        dot = "." if module_str else ""
        return f"{module_str}{dot}{op_name}"


@cartesian_params(api_variants, module_kinds)
def test_module(api_module_op, kind):
    api, module, op = api_module_op

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="module", kind=kind)

    p = pipe()
    p.build()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    expected = baseline_module_path(api, module, kind)
    assert out_str == expected, f"Expected {expected}, got {out_str}"


@cartesian_params(api_variants, module_kinds)
def test_op_name(api_module_op, kind):
    api, module, op = api_module_op

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="op_name", kind=kind)

    p = pipe()
    p.build()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    expected = baseline_display_name(api, module, kind)
    assert out_str == expected, f"Expected {expected}, got {out_str}"
