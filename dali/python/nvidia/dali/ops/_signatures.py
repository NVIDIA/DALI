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

from typing import Union, Optional
from typing import List, Set, Dict, Tuple, Any

from nvidia.dali import backend as _b
from nvidia.dali import types as _types
from nvidia.dali.ops import _registry, _names
from nvidia.dali.fn import _to_snake_case

from nvidia.dali.data_node import DataNode

_MAX_INPUT_SPELLED_OUT = 5


def _scalar_element_annotation(scalar_dtype):
    # TODO(klecki): Do we care for proper scalar types?
    # TODO(klecki): provide non-hacky implementation
    # now we just abuse the conversion and find a type
    conv_fn = _types._known_types[scalar_dtype][1]
    try:
        dummy_val = conv_fn(0)
        t = type(dummy_val)
        return t
    except:
        return Any


def _arg_type_annotation(arg_dtype):
    """Whatever goes as "scalar" argument for DALI Schema (as opposed to an argument input)

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


def _call_signature(schema, include_self=False, add_kwargs=False):
    input_list = []
    if include_self:
        input_list.append(inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD))
    if schema.HasInputDox():
        for i in range(schema.MinNumInput()):
            input_list.append(
                inspect.Parameter(f"__{schema.GetInputName(i)}",
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD))
        for i in range(schema.MinNumInput(), schema.MaxNumInput()):
            input_list.append(
                inspect.Parameter(f"__{schema.GetInputName(i)}",
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None))
    if not schema.HasInputDox():
        if schema.MaxNumInput() > _MAX_INPUT_SPELLED_OUT:
            input_list.append(
                inspect.Parameter("input", inspect.Parameter.VAR_POSITIONAL, annotation=DataNode))
        else:
            for i in range(schema.MinNumInput()):
                input_list.append(
                    inspect.Parameter(f"__input_{i}", inspect.Parameter.POSITIONAL_ONLY,
                                      annotation=DataNode))
            for i in range(schema.MinNumInput(), schema.MaxNumInput()):
                input_list.append(
                    inspect.Parameter(f"__input_{i}", inspect.Parameter.POSITIONAL_ONLY,
                                      default=None, annotation=DataNode))
    if add_kwargs:
        for arg in schema.GetArgumentNames():
            if schema.IsDeprecatedArg(arg):
                # We don't put the deprecated args in the visible API
                continue
            arg_dtype = schema.GetArgumentType(arg)
            # scalar_type = int  # _types._type_name_convert_to_string(arg_dtype, allow_tensors=False)
            scalar_type =  _arg_type_annotation(arg_dtype)
            is_arg_input = schema.IsTensorArgument(arg)

            annotation = Union[DataNode, scalar_type] if is_arg_input else scalar_type

            # providing any default changes DALI semantics
            input_list.append(
                Parameter(name=arg, kind=Parameter.KEYWORD_ONLY, default=Parameter.empty,
                          annotation=annotation))
    input_list.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(input_list)


def gen_all_signatures():
    for schema_name in _registry._all_registered_ops():
        schema = _b.TryGetSchema(schema_name)
        if schema is None:
            continue
        if schema.IsDocHidden():
            continue
        dotted_name, module_path, op_name = _names._process_op_name(schema_name)
        fn_name = _to_snake_case(op_name)
        print(f"Converting {module_path}: {op_name}/{fn_name}:")
        print(_call_signature(schema, add_kwargs=True, include_self=False))