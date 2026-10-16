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

"""Inputs hidden from the dynamic (ndd) API (`nvidia.dali.ops._names.MERGED_INPUT_ARGS`): the
runtime call signature, the generated typing (used for the `.pyi` stubs) and the docs must agree,
or IDE-suggested calls fail at runtime.
"""

import functools
import inspect
from inspect import Parameter

import nvidia.dali.backend as _b
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nvidia.dali.ops import _docs, _names, _signatures

_VARIADIC = _names._get_variadic_input_name()


def _positional_inputs(schema_name, api):
    schema = _b.GetSchema(schema_name)
    sig = _signatures._call_signature(schema, api, include_kwargs=False)
    return [
        (p.name, p.kind)
        for p in sig.parameters.values()
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL)
    ]


_MERGED = [
    ("Slice", ["anchor", "shape_input"]),
    ("decoders__ImageSlice", ["anchor", "shape_input"]),
    ("experimental__decoders__ImageSlice", ["input_1", "input_2"]),
    ("Reshape", ["shape_input"]),
    ("Reinterpret", ["shape_input"]),
]


@params(*_MERGED)
def test_dynamic_signature_hides_merged_inputs_but_keeps_catchall(schema_name, merged):
    """The hidden inputs are still accepted positionally (see
    `dynamic._op_builder._filter_merged_inputs`), so the typing keeps a positional catch-all."""
    inputs = _positional_inputs(schema_name, "dynamic")
    names = [name for name, _ in inputs]
    for name in merged:
        assert name not in names, f"{name!r} still advertised in {names}"
    assert inputs[0][1] == Parameter.POSITIONAL_ONLY  # the data input ("input_0" in the alias)
    assert inputs[-1] == (_VARIADIC, Parameter.VAR_POSITIONAL)
    assert len(inputs) == 2


@params(*_MERGED)
def test_fn_signature_keeps_merged_inputs(schema_name, merged):
    """The fn/ops APIs are unaffected by the dynamic-API-only merge."""
    for api in ("fn", "ops"):
        names = [name for name, _ in _positional_inputs(schema_name, api)]
        for name in merged:
            assert name in names, f"{name!r} missing from {api} signature {names}"


def _inputs_doc(schema_name, api):
    return _docs._get_inputs_doc(_b.GetSchema(schema_name), api)


def _resolve(root, dotted_name):
    return functools.reduce(getattr, dotted_name.split("."), root)


def _runtime_positional_inputs(func):
    return [
        (p.name, p.kind)
        for p in inspect.signature(func).parameters.values()
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL) and p.name != "self"
    ]


@params(*_MERGED)
def test_runtime_signature_matches_typing(schema_name, merged):
    """The runtime signatures of the fn-style function and of the operator class's `__call__`
    (`dynamic._op_builder._filter_merged_inputs`) hide the merged inputs the same way as the
    generated typing (`_signatures._filter_merged_input_params`)."""
    expected = _positional_inputs(schema_name, "dynamic")
    function = _resolve(ndd, _names._op_name(schema_name, "dynamic"))
    op_class = _resolve(ndd._ops, _names._op_name(schema_name, "ops"))
    for func in (function, op_class.__call__):
        actual = _runtime_positional_inputs(func)
        assert actual == expected, f"{func.__qualname__}: {actual} != {expected}"
        for name in merged:
            assert name not in [n for n, _ in actual]


@params(*_MERGED)
def test_dynamic_docs_describe_merged_inputs_as_catchall(schema_name, merged):
    """The ndd docs document the hidden inputs as the `*inputs` entry of the signature, not as
    separate parameters, and say which named arguments they stand for."""
    doc = _inputs_doc(schema_name, "dynamic")
    assert doc.count(f"*{_VARIADIC} :") == 1, doc
    assert f"{_VARIADIC}[" not in doc, doc  # no generic entry for the same inputs
    for name in merged:
        assert f"{name} :" not in doc, doc
    for name, args in _names.get_merged_inputs(schema_name).items():
        instead_of = " / ".join(f"`{arg}`" for arg in args)
        assert f"``{name}`` (instead of {instead_of})" in doc, doc


@params("Slice", "decoders__ImageSlice")
def test_dynamic_docs_keep_slice_input_semantics(schema_name):
    """Slice's positional inputs are interpreted by their dtype, unlike the named arguments, so
    their input docs are kept in the `*inputs` entry (`normalized_anchor` refers to them)."""
    doc = _inputs_doc(schema_name, "dynamic")
    assert "``anchor`` (1D Tensor/Batch of float or int):" in doc, doc
    assert "``shape_input`` (1D Tensor/Batch of float or int):" in doc, doc
    assert "given together" in doc, doc
    assert "`normalized_anchor`" in doc and "`normalized_shape`" in doc, doc


@params(
    ("Slice", ["anchor", "shape_input"]),
    ("Reshape", ["shape_input"]),
    ("Reinterpret", ["shape_input"]),
)
def test_fn_docs_keep_merged_inputs(schema_name, merged):
    doc = _inputs_doc(schema_name, "fn")
    for name in merged:
        assert f"{name} :" in doc, doc


def test_warp_affine_gpu_matrix_input_is_not_merged():
    """WarpAffine's `mtx` input is consumed on the GPU, unlike the merged inputs above, and is the
    only way to keep the matrix on the GPU (arguments are always CPU). It stays a visible input."""
    assert not _names.get_merged_input_names("WarpAffine")
    names = [name for name, _ in _positional_inputs("WarpAffine", "dynamic")]
    assert "mtx" in names, names
    assert "mtx :" in _inputs_doc("WarpAffine", "dynamic")
