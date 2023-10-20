# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn as _functional


def _schema_name(cls):
    """Extract the name of the schema from Operator class."""
    return getattr(cls, 'schema_name', cls.__name__)


def _process_op_name(op_schema_name, make_hidden=False):
    """Based on the schema name (for example "Resize" or "experimental__readers__Video")
    transform it into Python-compatible module & operator name information.

    Parameters
    ----------
    op_schema_name : str
        The name of the schema
    make_hidden : bool, optional
        Should a .hidden module be added to the module path to indicate an internal operator,
        that it's later reimported but not directly discoverable, by default False

    Returns
    -------
    (str, list, str)
        (Full name with all submodules, submodule path to the operator, name of the operator),
        for example:
            ("Resize", [], "Resize") or
            ("experimental.readers.Video", ["experimental", "readers"], "Video")
    """
    # Two underscores (reasoning: we might want to have single underscores in the namespace itself)
    namespace_delim = "__"
    op_full_name = op_schema_name.replace(namespace_delim, '.')
    *submodule, op_name = op_full_name.split('.')
    if make_hidden:
        submodule = [*submodule, 'hidden']
    return op_full_name, submodule, op_name


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
        API type, "ops" or "fn", by default "fn"

    Returns
    -------
    str
        The fully qualified name in given API
    """
    full_name, submodule, op_name = _process_op_name(op_schema_name)
    if api == "fn":
        return ".".join([*submodule, _functional._to_snake_case(op_name)])
    elif api == "ops":
        return full_name
    else:
        raise ValueError(f'{api} is not a valid DALI api name, try one of {"fn", "ops"}')
