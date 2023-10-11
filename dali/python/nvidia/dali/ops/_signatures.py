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

import inspect
from inspect import Parameter, Signature
import typing
import os

from pathlib import Path

from typing import Union, Optional
from typing import List, Any

from nvidia.dali import backend as _b
from nvidia.dali import types as _types
from nvidia.dali.ops import _registry, _names, _docs
from nvidia.dali.fn import _to_snake_case

from nvidia.dali.data_node import DataNode

_MAX_INPUT_SPELLED_OUT = 5


def _scalar_element_annotation(scalar_dtype):
    # TODO(klecki): provide non-hacky implementation
    # now we just abuse the conversion and find a type, doesn't work for TFRecord
    conv_fn = _types._known_types[scalar_dtype][1]
    try:
        dummy_val = conv_fn(0)
        t = type(dummy_val)
        return t
    except:
        return Any


def _arg_type_annotation(arg_dtype):
    """Convert regular key-word argument type to annotation. Handles Lists and scalars.

    Parameters
    ----------
    arg_dtype : _type_
        _description_
    """
    if arg_dtype in _types._vector_types:
        scalar_dtype = _types._vector_types[arg_dtype]
        scalar_annotation = _scalar_element_annotation(scalar_dtype)
        return Union[List[scalar_annotation], scalar_annotation]
    return _scalar_element_annotation(arg_dtype)


def _get_positional_input_param(schema, idx):
    # Only first MinNumInputs are mandatory, the rest are optional:
    default = Parameter.empty if idx < schema.MinNumInput() else None
    annotation = DataNode if idx < schema.MinNumInput() else Optional[DataNode]
    if schema.HasInputDox():
        return Parameter(f"__{schema.GetInputName(idx)}", kind=Parameter.POSITIONAL_ONLY,
                         default=default, annotation=annotation)
    else:
        return Parameter(f"__input_{idx}", kind=Parameter.POSITIONAL_ONLY, default=default,
                         annotation=annotation)

def _get_positional_input_params(schema):
    param_list = []
    if schema.MaxNumInput() > _MAX_INPUT_SPELLED_OUT:
        param_list.append(Parameter("input", Parameter.VAR_POSITIONAL, annotation=DataNode))
    else:
        for i in range(schema.MaxNumInput()):
            param_list.append(_get_positional_input_param(schema, i))
    return param_list

def _get_keyword_params(schema):
    param_list = []
    for arg in schema.GetArgumentNames():
        if schema.IsDeprecatedArg(arg):
            # We don't put the deprecated args in the visible API
            continue
        arg_dtype = schema.GetArgumentType(arg)
        scalar_type =  _arg_type_annotation(arg_dtype)
        is_arg_input = schema.IsTensorArgument(arg)

        annotation = Union[DataNode, scalar_type] if is_arg_input else scalar_type
        if schema.IsArgumentOptional(arg):
            annotation = Optional[annotation]

        # TODO(klecki): What to do with the defaults?
        param_list.append(
            Parameter(name=arg, kind=Parameter.KEYWORD_ONLY, default=Parameter.empty,
                      annotation=annotation))
    # We always have the **kwargs
    param_list.append(Parameter("kwargs", Parameter.VAR_KEYWORD))
    return param_list

def _call_signature(schema, include_inputs=True, include_kwargs=True, include_self=False):
    param_list = []
    if include_self:
        # TODO(klecki): what kind of parameter is `self`?
        param_list.append(Parameter("self", kind=Parameter.POSITIONAL_ONLY))

    if include_inputs:
        param_list.extend(_get_positional_input_params(schema))

    if include_kwargs:
        param_list.extend(_get_keyword_params(schema))
    return inspect.Signature(param_list)

# TODO(klecki): generate return type?
def _gen_fn_signature(schema, schema_name, fn_name):
    return f"""
def {fn_name}{_call_signature(schema, include_inputs=True, include_kwargs=True)}:
    \"""{_docs._docstring_generator_fn(schema_name)}
    \"""
    ...
"""

def _gen_ops_signature(schema, schema_name, cls_name):
    return f"""
class {cls_name}:
    \"""{_docs._docstring_generator(schema_name)}
    \"""
    def __init__{_call_signature(schema, include_inputs=False, include_kwargs=True, include_self=True)}:
        ...

    def __call__{_call_signature(schema, include_inputs=True, include_kwargs=True, include_self=True)}:
        \"""{_docs._docstring_generator_call(schema_name)}
        \"""
        ...
"""


_HEADER = """
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

from typing import Union, Optional
from typing import List, Any

from nvidia.dali.data_node import DataNode
"""


def _build_module_tree():
    module_tree = {}
    processed = set()
    for schema_name in _registry._all_registered_ops():
        schema = _b.TryGetSchema(schema_name)
        if schema is None:
            continue
        if schema.IsDocHidden() or schema.IsInternal():
            continue
        dotted_name, module_nesting, op_name = _names._process_op_name(schema_name)
        if dotted_name not in processed:
            module_nesting.insert(0, "")  # add the top-level module
            curr_dict = module_tree
            # add all submodules on the path
            for curr_module in module_nesting:
                if curr_module not in curr_dict:
                    curr_dict[curr_module] = dict()
                curr_dict = curr_dict[curr_module]
    return module_tree


# TODO(klecki): Generate the full hierarchy of submodules, each higher level module
# needs to import all child modules as `from child_module import *` ?
# or just use the `from . import child_module`  <- I think the latter!!!
def gen_all_signatures(whl_path, api):
    whl_path = Path(whl_path)
    module_tree = _build_module_tree()
    module_to_file = {}
    for schema_name in _registry._all_registered_ops():
        schema = _b.TryGetSchema(schema_name)
        if schema is None:
            continue
        if schema.IsDocHidden() or schema.IsInternal():
            continue
        dotted_name, module_nesting, op_name = _names._process_op_name(schema_name)
        fn_name = _to_snake_case(op_name)
        module_path = Path("/".join(module_nesting))

        if module_path not in module_to_file:
            file_path = whl_path / api / module_path / "__init__.py"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, "w").close()  # clear the file
            f = open(file_path, "a")
            module_to_file[module_path] = f
            f.write(_HEADER)
            full_module_nesting = [""] + module_nesting
            print(f"{full_module_nesting=}")
            submodules_dict = module_tree
            for submodule in full_module_nesting:
                submodules_dict = submodules_dict[submodule]
            direct_submodules = submodules_dict.keys()
            for direct_submodule in direct_submodules:
                f.write(f"from . import {direct_submodule}\n")

            f.write("\n\n")

        if api == "fn":
            module_to_file[module_path].write(_gen_fn_signature(schema, schema_name, fn_name))
        else:
            module_to_file[module_path].write(_gen_ops_signature(schema, schema_name, op_name))

        # print(f"Converting {module_path}: {op_name}/{fn_name}:")
        # print(_call_signature(schema, include_kwargs=True, include_self=False))

    for _, f in module_to_file.items():
        f.close()
