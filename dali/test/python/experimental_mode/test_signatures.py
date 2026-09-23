# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Signature generation (used for the dynamic-API `.pyi` stubs) must stay in sync with
`nvidia.dali.ops._names.MERGED_INPUT_ARGS` / `MERGED_MULTI_INPUT_ARGS`: an input hidden from the
dynamic (ndd) API's runtime call signature must not still be advertised as a valid parameter in
the generated typing, or IDE-suggested calls fail at runtime.
"""

from inspect import Parameter

import nvidia.dali.backend as _b
from nvidia.dali.ops import _signatures


def _positional_input_names(schema_name, api):
    schema = _b.GetSchema(schema_name)
    sig = _signatures._call_signature(schema, api, include_kwargs=False)
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL)
    ]


def test_warp_affine_dynamic_signature_hides_merged_input_but_keeps_catchall():
    # "mtx" is merged into the "matrix" argument for the dynamic API...
    names = _positional_input_names("WarpAffine", "dynamic")
    assert "mtx" not in names
    # ...but WarpAffine's merge is GPU-routable, so a catch-all must remain so a GPU-placed
    # `matrix` (or a plain positional `mtx`, for backward compatibility) can still be given
    # positionally at runtime - see `dynamic._op_builder._filter_merged_inputs`.
    assert names[-1] == _signatures._names._get_variadic_input_name()


def test_warp_affine_fn_signature_keeps_named_input():
    # The fn/ops APIs are unaffected by the dynamic-API-only merge.
    names = _positional_input_names("WarpAffine", "fn")
    assert "mtx" in names


def test_reshape_dynamic_signature_hides_merged_input_without_catchall():
    # "shape_input" is merged into the "shape" argument for the dynamic API. Unlike WarpAffine,
    # this merge isn't GPU-routable, so no caller can reach it positionally any more; the
    # generated signature must not advertise it as if they still could.
    names = _positional_input_names("Reshape", "dynamic")
    assert "shape_input" not in names
    assert _signatures._names._get_variadic_input_name() not in names
