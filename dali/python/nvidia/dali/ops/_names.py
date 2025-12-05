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
