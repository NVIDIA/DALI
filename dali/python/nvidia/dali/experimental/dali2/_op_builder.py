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
from ._batch import Batch
from ._tensor import Tensor
import warnings
from . import ops
from . import fn
import types
import copy
import sys
from . import _invocation, _device
import nvidia.dali.ops as _ops


def _is_tensor_type(x, nested_list_warning=False):
    if isinstance(x, Batch):
        raise ValueError("A list of Batchs is not a valid argument type")
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
            "not a list of 1D tensors. Convert the list to a Tensor, Batch or "
            "a list of Tensors to avoid this warning."
        )
    return False


def _is_batch(x):
    if isinstance(x, Batch):
        return True
    if isinstance(x, (_b.TensorListCPU, _b.TensorListGPU)):
        return Batch(x)
    if isinstance(x, list):
        return any(_is_tensor_type(t, True) for t in x)
    return False


def _to_tensor(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, _invocation.InvocationResult):
        if x.is_batch:
            raise ValueError("Batch invocation result cannot be used as a single tensor")
        return Tensor(invocation_result=x)
    return Tensor(x)


def _to_batch(x, batch_size):
    if isinstance(x, Batch):
        return x
    if isinstance(x, _invocation.InvocationResult):
        if x.is_batch:
            return Batch(invocation_result=x)
        else:
            x = _to_tensor(x)  # fall back to regular replication
    if _is_batch(x):
        return Batch(x)

    return Batch([_to_tensor(x)] * batch_size)


_unsupported_args = {"bytes_per_sample_hint", "preserve"}


def _find_or_create_module(root_module, module_path):
    module = root_module
    for path_part in module_path:
        module = getattr(module, path_part, None)
        if module is None:
            module = types.ModuleType(path_part)
            setattr(root_module, path_part, module)
    return module


def build_operator_class(schema):
    class_name = schema.OperatorName()
    module_path = schema.ModulePath()
    module = ops
    legacy_op_class = None
    import nvidia.dali.ops

    legacy_op_module = nvidia.dali.ops
    for path_part in module_path:
        legacy_op_module = getattr(legacy_op_module, path_part)
    module = _find_or_create_module(module, module_path)

    legacy_op_class = getattr(legacy_op_module, class_name)
    op_class = type(class_name, (ops.Operator,), {})
    op_class.schema = schema
    op_class.op_name = class_name
    op_class.fn_name = _to_snake_case(class_name)
    op_class.legacy_op = legacy_op_class
    op_class._instance_cache = {}
    op_class.__init__ = build_constructor(schema, op_class)
    op_class.__call__ = build_call_function(schema, op_class)
    op_class.__module__ = module.__name__
    op_class.__qualname__ = class_name
    setattr(module, class_name, op_class)
    return op_class


def build_constructor(schema, op_class):
    stateful = schema.IsStateful()
    function_name = "__init__"

    call_args = []
    for arg in schema.GetArgumentNames():
        if arg in _unsupported_args:
            continue
        if schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
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
        if arg in _unsupported_args:
            continue
        if not schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
            call_args.append(f"{arg}=None")
        else:
            call_args.append(arg)

    inputs = []
    min_inputs = schema.MinNumInput()
    max_inputs = schema.MaxNumInput()
    input_indices = {}
    arguments = schema.GetArgumentNames()
    for i in range(max_inputs):
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
        inputs = inputs + ["/"]
    header = f"__call__({', '.join(['self'] + inputs + call_args)})"

    def call(self, *raw_args, batch_size=None, **raw_kwargs):
        is_batch = batch_size is not None
        if batch_size is None:
            for i, x in enumerate(list(raw_args) + list(raw_kwargs.values())):
                if _is_batch(x):
                    is_batch = True
                    if isinstance(x, Batch):
                        x_batch_size = x.batch_size
                    elif isinstance(x, list):
                        x_batch_size = len(x)
                    if batch_size is not None:
                        if x_batch_size != batch_size:
                            raise ValueError(
                                f"Inconsistent batch size: {x_batch_size} != {batch_size}"
                            )
                    else:
                        batch_size = x_batch_size
        if not is_batch:
            batch_size = self._max_batch_size
            is_batch = False

        inputs = []
        kwargs = {}

        if is_batch:
            for inp in raw_args:
                if inp is None:
                    continue
                inp = _to_batch(inp, batch_size)
                inputs.append(inp)
            for k, v in raw_kwargs.items():
                if v is None:
                    continue
                kwargs[k] = _to_batch(v, batch_size)
        else:
            for inp in raw_args:
                if inp is None:
                    continue
                inputs.append(_to_tensor(inp))
            for k, v in raw_kwargs.items():
                if v is None:
                    continue
                kwargs[k] = _to_tensor(v)

        inputs = [copy.copy(x) for x in inputs]
        kwargs = {k: copy.copy(v) for k, v in kwargs.items()}
        if stateful:
            call_id = self._call_id
            self._call_id += 1
        else:
            call_id = None
        invocation = _invocation.Invocation(
            self, call_id, inputs, kwargs, is_batch=is_batch, batch_size=batch_size
        )
        if is_batch:
            if len(invocation) == 1:
                return Batch(invocation_result=invocation[0])
            else:
                return Batch(invocation_result=invocation)
        else:
            if len(invocation) == 1:
                return Tensor(invocation_result=invocation[0])
            else:
                return Tensor(invocation_result=invocation)

    function = makefun.create_function(header, call)

    return function


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def build_fn_wrapper(op):
    schema = op.schema
    module_path = schema.ModulePath()
    module = fn
    for path_part in module_path:
        new_module = getattr(module, path_part, None)
        if new_module is None:
            new_module = types.ModuleType(path_part)
            setattr(module, path_part, new_module)
        module = new_module

    fn_name = _to_snake_case(op.schema.OperatorName())
    inputs = []
    min_inputs = schema.MinNumInput()
    max_inputs = schema.MaxNumInput()
    input_indices = {}
    arguments = schema.GetArgumentNames()
    for i in range(max_inputs):
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

    fixed_args = []
    tensor_args = []
    signature_args = []
    for arg in op.schema.GetArgumentNames():
        if arg in _unsupported_args:
            continue
        if op.schema.IsTensorArgument(arg):
            tensor_args.append(arg)
        else:
            fixed_args.append(arg)
        if op.schema.IsArgumentOptional(arg):
            signature_args.append(f"{arg}=None")
        else:
            signature_args.append(arg)

    if signature_args:
        signature_args = ["*"] + signature_args
    if inputs:
        inputs = inputs + ["/"]
    header = f"{fn_name}({', '.join(inputs + signature_args)})"

    def call(*inputs, batch_size=None, **raw_kwargs):
        max_batch_size = raw_kwargs.get("max_batch_size", batch_size)
        if max_batch_size is None:
            max_batch_size = 1
            for i, x in enumerate(inputs):
                if _is_batch(x):
                    if not isinstance(x, Batch):
                        x = Batch(x)
                    max_batch_size = max(max_batch_size, x.batch_size)
        max_batch_size = _next_pow2(max_batch_size)
        init_args = {
            arg: raw_kwargs[arg]
            for arg in fixed_args
            if arg != "max_batch_size" and arg in raw_kwargs
        }
        op_inst = op.get(None, max_batch_size=max_batch_size, **init_args)
        call_args = {arg: raw_kwargs[arg] for arg in tensor_args if arg in raw_kwargs}

        return op_inst(*inputs, **call_args)

    function = makefun.create_function(header, call)
    function.op_class = op
    function.schema = schema
    setattr(module, fn_name, function)
    return function


def build_operators():
    _all_ops = _ops._registry._all_registered_ops()
    all_op_classes = []
    deprecated = {}
    op_map = {}
    for op_name in _all_ops:
        if op_name.endswith("ExternalSource") or op_name.endswith("PythonFunction"):
            continue

        schema = _b.GetSchema(op_name)
        deprecated_in_favor = schema.DeprecatedInFavorOf()
        if deprecated_in_favor:
            deprecated[op_name] = deprecated_in_favor
        cls = build_operator_class(schema)
        all_op_classes.append(cls)
        op_map[op_name] = cls
    for deprecated, in_favor in deprecated.items():
        schema = _b.GetSchema(deprecated)
        module = _find_or_create_module(ops, schema.ModulePath())
        setattr(module, deprecated, op_map[in_favor])

    return all_op_classes
