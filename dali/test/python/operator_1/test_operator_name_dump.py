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


module_variants = [
    ("nvidia.dali.fn", fn.name_dump),
    ("nvidia.dali.fn.sub", fn.sub.name_dump),
    ("nvidia.dali.fn.sub.sub", fn.sub.sub.name_dump),
    ("nvidia.dali.ops", ops.NameDump()),
    ("nvidia.dali.ops.sub", ops.sub.NameDump()),
    ("nvidia.dali.ops.sub.sub", ops.sub.sub.NameDump()),
]


@params(*module_variants)
def test_api_name(module, op):

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="module")

    p = pipe()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    assert out_str == module, f"Expected {module}, got {out_str}"


def baseline_display_name(module, include_module):
    op_name = "name_dump" if "fn" in module else "NameDump"
    if not include_module:
        return op_name
    else:
        return f"{module}.{op_name}"


@cartesian_params(module_variants, [True, False])
def test_op_name(module_op, include_module):
    module, op = module_op

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return op(target="op_name", include_module=include_module)

    p = pipe()
    (out,) = p.run()
    out_str = extract_str_from_tl(out)
    expected = baseline_display_name(module, include_module)
    assert out_str == expected, f"Expected {expected}, got {out_str}"
