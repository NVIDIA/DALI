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
from nose2.tools import params
from nvidia.dali.ops import _names, _signatures

_VARIADIC = _names._get_variadic_input_name()


def _positional_inputs(schema_name, api):
    schema = _b.GetSchema(schema_name)
    sig = _signatures._call_signature(schema, api, include_kwargs=False)
    return [
        (p.name, p.kind)
        for p in sig.parameters.values()
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL)
    ]


@params(
    ("WarpAffine", ["mtx"]),
    ("Slice", ["anchor", "shape_input"]),
)
def test_dynamic_signature_hides_merged_inputs_but_keeps_catchall(schema_name, merged):
    """GPU-routable (WarpAffine) and multi-input (Slice) merges keep a positional catch-all at
    runtime (see `dynamic._op_builder._filter_merged_inputs`), so the typing must too."""
    inputs = _positional_inputs(schema_name, "dynamic")
    names = [name for name, _ in inputs]
    for name in merged:
        assert name not in names, f"{name!r} still advertised in {names}"
    assert inputs[0] == ("data", Parameter.POSITIONAL_ONLY)
    assert inputs[-1] == (_VARIADIC, Parameter.VAR_POSITIONAL)
    assert len(inputs) == 2


@params("Reshape", "Reinterpret")
def test_dynamic_signature_hides_merged_input_without_catchall(schema_name):
    """Non-GPU-routable single-input merges drop the positional slot at runtime; the typing
    must not advertise it (neither by name nor through a catch-all)."""
    inputs = _positional_inputs(schema_name, "dynamic")
    assert inputs == [("data", Parameter.POSITIONAL_ONLY)]


@params(
    ("WarpAffine", ["mtx"]),
    ("Slice", ["anchor", "shape_input"]),
    ("Reshape", ["shape_input"]),
    ("Reinterpret", ["shape_input"]),
)
def test_fn_signature_keeps_merged_inputs(schema_name, merged):
    """The fn/ops APIs are unaffected by the dynamic-API-only merge."""
    for api in ("fn", "ops"):
        names = [name for name, _ in _positional_inputs(schema_name, api)]
        for name in merged:
            assert name in names, f"{name!r} missing from {api} signature {names}"
