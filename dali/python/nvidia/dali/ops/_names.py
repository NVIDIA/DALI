# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import backend as _b
from nvidia.dali import fn as _functional


def _schema_name(cls):
    """Extract the name of the schema from Operator class."""
    return getattr(cls, "schema_name", cls.__name__)


def _process_op_name(op_schema_name, make_hidden=False, api="ops"):
    """Based on the schema name (for example "Resize" or "experimental__readers__Video")
    transform it into Python-compatible module & operator name information.

    Parameters
    ----------
    op_schema_name : str
        The name of the schema
    make_hidden : bool, optional
        Should a .hidden module be added to the module path to indicate an internal operator,
        that it's later reimported but not directly discoverable, by default False
    api : str, optional
        API type, "ops", "fn", or "dynamic", by default "ops"

    Returns
    -------
    (str, list, str)
        (Full name with all submodules, submodule path to the operator, name of the operator),
        for example:
            ("Resize", [], "Resize") or
            ("experimental.readers.Video", ["experimental", "readers"], "Video")
    """

    schema = _b.GetSchema(op_schema_name)
    submodule_path = schema.ModulePath()
    op_name = schema.OperatorName()
    if make_hidden:
        submodule_path = [*submodule_path, "hidden"]

    to_snake_case = False
    if api == "fn":
        to_snake_case = True
    elif api == "dynamic":
        if not submodule_path:
            to_snake_case = True
        elif submodule_path[0] == "ops":
            to_snake_case = False
        elif "readers" in submodule_path:
            to_snake_case = False
        else:
            to_snake_case = True

    if to_snake_case:
        op_name = _functional._to_snake_case(op_name)
    op_full_name = ".".join(submodule_path + [op_name])
    return op_full_name, submodule_path, op_name


def _op_name(op_schema_name, api="fn"):
    """Extract the name of the operator from the schema and return it transformed for given API:
    CamelCase for "ops" API, and snake_case for "fn" API. The name contains full module path,
    for example:
        _op_name("experimental__readers__VideoResize", "fn") -> "experimental.readers.video_resize"

    Parameters
    ----------
    op_schema_name : str
        The name of the schema
    api : str, optional
        API type, "ops", "fn", or "dynamic", by default "fn"

    Returns
    -------
    str
        The fully qualified name in given API
    """
    full_name, _, _ = _process_op_name(op_schema_name, api=api)
    return full_name


def _get_input_name(schema, input_idx):
    """Return the string representing the name of positional-only input to the operator.
    This function appends the double underscore `__`, that indicates via the mypy convention,
    that all inputs are positional-only. This happens also for the names introduced via schema.

    Parameters
    ----------
    schema : OpSchema
        schema to query
    input_idx : int
        Index of the input
    """
    if schema.HasInputDox():
        name = schema.GetInputName(input_idx)
    elif schema.MaxNumInput() == 1:
        name = "input"
    else:
        name = f"input_{input_idx}"
    # Add "_input" at the end, if the name doesn't already contain "input" or prepend "__".
    # Keep adding underscores until there's no name clash
    while schema.HasArgument(name):
        if "input" in name:
            name = "__" + name
        else:
            name += "_input"
    return name


def _get_variadic_input_name():
    """Return the string representing the name of positional-only input for a variadic context."""
    return "inputs"


# Mode spec, "Uniform inputs and arguments": some operators have a positional input and an
# optional argument that represent the same value, differing only in device placement, e.g.
# Reshape's `shape_input` (CPU-only) vs. its `shape` tensor-argument (also always CPU). Exposing
# both separately in the dynamic (ndd) API is redundant, so for that API the input is hidden and
# the argument is the single canonical, merged parameter for that value. The fn/ops APIs are
# unaffected and keep exposing both, as before.
MERGED_INPUT_ARGS = {
    "Reshape": "shape_input",
    "Reinterpret": "shape_input",
    "WarpAffine": "mtx",
}

# Same idea as `MERGED_INPUT_ARGS`, but for operators where a single positional input is
# redundant with more than one named argument at once (the named arguments being mutually
# exclusive with each other, so at most one is ever actually given). Slice's `anchor` input
# is equivalent to giving either `start` (absolute) or `rel_start` (relative); its `shape`
# input (named `shape_input` once disambiguated from the `shape` argument, see
# `_get_input_name`) is equivalent to giving either `shape` or `rel_shape`. `end`/`rel_end`
# have no positional-input equivalent at all, so they are unaffected either way.
#
# Unlike WarpAffine's `mtx`/`matrix`, there is no GPU-routing fallback here: Slice's own doc
# already discourages placing `anchor`/`shape` on GPU (it costs an extra D2H copy), and,
# critically, the operator requires `anchor` and `shape` to be given together as positional
# inputs or not at all (never just one) - so a GPU value for only one of the merged argument
# groups can't be routed as a lone extra positional input the way WarpAffine's `matrix` is.
MERGED_MULTI_INPUT_ARGS = {
    "Slice": {
        "anchor": ("start", "rel_start"),
        "shape_input": ("shape", "rel_shape"),
    },
}

# Subset of `MERGED_INPUT_ARGS` whose hidden input allowed GPU placement
# (InputDevice::MatchBackendOrCPU rather than InputDevice::CPU), mapped to the merged argument
# name. Since arguments must always be CPU (see Mode spec, "Special arguments"), a GPU-placed
# value passed for that argument is routed through as a positional input instead, in the dynamic
# API runtime (see `dynamic._op_builder._route_gpu_merged_arg`); typing/signature generation
# (`_signatures._get_positional_input_params`) also consults this, via
# `merge_restores_catchall`, to keep the generated signature able to accept it positionally.
MERGED_ARG_GPU_INPUT = {
    "WarpAffine": "matrix",
}


def merge_restores_catchall(schema_name):
    """Whether hiding `schema_name`'s merged input(s) (see `get_merged_input_names`) from a
    generated signature must keep a positional catch-all in place.

    GPU-routable (`MERGED_ARG_GPU_INPUT`) and multi-input (`MERGED_MULTI_INPUT_ARGS`) merged
    cases were reachable positionally before the merge and must stay that way; see
    `dynamic._op_builder._filter_merged_inputs` for the full rationale.
    """
    return schema_name in MERGED_ARG_GPU_INPUT or schema_name in MERGED_MULTI_INPUT_ARGS


def get_merged_input_names(schema_name):
    """Return the set of dynamic-API-hidden input names for `schema_name` (empty if none).

    Combines `MERGED_INPUT_ARGS` (1 input <-> 1 argument) and `MERGED_MULTI_INPUT_ARGS`
    (1 input <-> several mutually exclusive arguments).
    """
    names = set()
    single = MERGED_INPUT_ARGS.get(schema_name)
    if single is not None:
        names.add(single)
    names.update(MERGED_MULTI_INPUT_ARGS.get(schema_name, {}))
    return names
