# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.backend as _b
from nvidia.dali.fn import _to_snake_case
import makefun
from ._tensor_list import TensorList
from ._tensor import Tensor
import warnings
from . import ops
import types
import copy
from . import _invocation
import nvidia.dali.ops as _ops


def _is_tensor_type(x, nested_list_warning=False):
    if isinstance(x, TensorList):
        raise ValueError("A list of TensorLists is not a valid argument type")
    if isinstance(x, Tensor):
        return True
    if hasattr(x, "__array__"):
        return True
    if hasattr(x, "__cuda_array_interface__"):
        return True
    if hasattr(x, "__dlpack__"):
        return True
    if nested_list_warning and isinstance(x, list):
        warnings.warn(
            "A nested list is ambiguous. It is interpreted as a single tensor, "
            "not a list of 1D tensors. Convert the list to a Tensor, TensorList or "
            "a list of Tensors to avoid this warning."
        )
    return False


def _is_batch(x):
    if isinstance(x, TensorList):
        return True
    if isinstance(x, list):
        return any(_is_tensor_type(t, True) for t in x)
    return False


def build_operator_class(schema):
    class_name = schema.OperatorName()
    module_path  = schema.ModulePath()
    module = ops
    legacy_op_class = None
    import nvidia.dali.ops
    legacy_op_module = nvidia.dali.ops
    print(module_path, class_name)
    for path_part in module_path:
        legacy_op_module = getattr(legacy_op_module, path_part)
        print(legacy_op_module)
        new_module = getattr(module, path_part, None)
        if new_module is None:
            new_module = types.ModuleType(path_part)
            setattr(module, path_part, new_module)
        module = new_module
    legacy_op_class = getattr(legacy_op_module, class_name)
    op_class = type(class_name, (ops.Operator,), {})
    op_class.schema = schema
    op_class.op_name = class_name
    op_class.fn_name = _to_snake_case(class_name)
    op_class.legacy_op = legacy_op_class
    op_class.__init__ = build_constructor(schema, op_class)
    op_class.__call__ = build_call_function(schema, op_class)
    op_class.__module__ = module.__name__
    op_class.__qualname__ = class_name
    module.__setattr__(class_name, op_class)
    return op_class


def build_constructor(schema, op_class):
    stateful = schema.IsStateful()
    function_name = "__init__"

    call_args = []
    for arg in schema.GetArgumentNames():
        if schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
            if schema.HasArgumentDefaultValue(arg):
                call_args.append(f"{arg}={schema.GetArgumentDefaultValueString(arg)}")
            else:
                call_args.append(f"{arg}=None")
        else:
            call_args.append(arg)

    if call_args:
        call_args = ["*"] + call_args
    header = f"__init__({', '.join(['self', 'max_batch_size', 'name=None'] + call_args)})"

    def init(self, max_batch_size, name, **kwargs):
        ops.Operator.__init__(self, max_batch_size, name, **kwargs)
        if stateful:
            self._call_id = 0

    function = makefun.create_function(header, init)
    function.__qualname__ = f"{op_class.__name__}.{function_name}"

    return function


def build_call_function(schema, op_class):
    stateful = schema.IsStateful()
    call_args = []
    for arg in schema.GetArgumentNames():
        if not schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
            if schema.HasArgumentDefaultValue(arg):
                call_args.append(f"{arg}={schema.GetArgumentDefaultValueString(arg)}")
            else:
                call_args.append(f"{arg}=None")
        else:
            call_args.append(arg)

    inputs = []
    min_inputs = schema.MinNumInput()
    max_inputs = schema.MaxNumInput()
    input_indices = {}
    arguments = schema.GetArgumentNames()
    for i in range(min_inputs, max_inputs):
        if schema.HasInputDox():
            input_name = schema.GetInputName(i)
            if input_name in arguments:
                input_name += "_input"
        else:
            input_name = f"input_{i}"
        input_indices[input_name] = i
        if i < min_inputs:
            inputs.append(f"{input_name}")
        else:
            inputs.append(f"{input_name}=None")

    if call_args:
        call_args = ["*"] + call_args
    if inputs:
        inputs = ["/"] + inputs
    header = f"__call__({', '.join(['self'] + inputs + call_args)})"

    def call(self, *args, batch_size=None, **kwargs):
        inputs = list(args)
        is_batch = batch_size is not None
        if batch_size is None:
            for i, x in enumerate(inputs + list(kwargs.values())):
                if _is_batch(x):
                    is_batch = True
                    break
        if not is_batch:
            batch_size = self._max_batch_size
            is_batch = True

        args = [copy.copy(x) for x in args]
        kwargs = {k: copy.copy(v) for k, v in kwargs.items()}
        if stateful:
            call_id = self._call_id
            self._call_id += 1
        else:
            call_id = None
        expr = _invocation.Invocation(self, call_id, args, kwargs, is_batch=is_batch, batch_size=batch_size)
        if is_batch:
            return TensorList(expression=expr)
        else:
            return Tensor(expression=expr)

    function = makefun.create_function(header, call)

    return function


def build_operators():
    _all_ops = _ops._registry._all_registered_ops()
    for op_name in _all_ops:
        if op_name.endswith("ExternalSource") or op_name.endswith("PythonFunction"):
            continue

        schema = _b.GetSchema(op_name)
        build_operator_class(schema)
