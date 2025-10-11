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
from ._batch import Batch, _get_batch_size
from ._tensor import Tensor
from . import ops
from . import _type
import types
import copy
from . import _invocation, _device, _eval_mode, _eval_context
import nvidia.dali.ops as _ops
import nvidia.dali.types
import nvtx


def is_external(x):
    if isinstance(x, Tensor):
        return x._is_external()
    if isinstance(x, Batch):
        return x._is_external()
    return False


def _scalar_decay(x):
    if isinstance(x, _device.Device):
        return x.device_type
    if isinstance(x, _type.DType):
        return x.type_id
    if x is str:
        return nvidia.dali.types.STRING
    if x is bool:
        return nvidia.dali.types.BOOL
    if x is int or x is float:
        raise ValueError(
            f"Do not use Python built-in type {x} as an argument. "
            f"Use one of the DALI types instead."
        )
    return x


def _get_input_device(x):
    with nvtx.annotate("get_input_device", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Batch):
            return x.device
        if isinstance(x, Tensor):
            return x.device
        if isinstance(x, _b.TensorListCPU):
            return _device.Device("cpu")
        if isinstance(x, _b.TensorListGPU):
            return _device.Device("gpu")
        if hasattr(x, "__cuda_array_interface__"):
            return _device.Device("gpu")
        if hasattr(x, "__dlpack_device__"):
            dev = x.__dlpack_device__()
            if int(dev[0]) == 1 or int(dev[0]) == 3:  # CPU or CPU_PINNED
                return _device.Device("cpu")
            elif int(dev[0]) == 2:
                return _device.Device("gpu", dev[1])
            else:
                raise ValueError(f"Unknown DLPack device type: {dev.type}")
        if hasattr(x, "__dlpack__"):
            return _device.Device("cpu")
        if isinstance(x, list) and x:
            return _get_input_device(x[0])
        return None


def _get_input_device_type(x):
    dev = _get_input_device(x)
    return dev.device_type if dev is not None else None


def _to_tensor(x, device=None):
    with nvtx.annotate("to_tensor", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Tensor):
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                raise ValueError("Batch invocation result cannot be used as a single tensor")
            return Tensor(invocation_result=x, device=device)
        return Tensor(x, device=device)


def _to_batch(x, batch_size, device=None):
    with nvtx.annotate("to_batch", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Batch):
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                return Batch(invocation_result=x, device=device)
            else:
                x = _to_tensor(x)  # fall back to regular replication
        actual_batch_size = _get_batch_size(x)
        if actual_batch_size is not None:
            if batch_size is not None and actual_batch_size != batch_size:
                raise ValueError(f"Unexpected batch size: {actual_batch_size} != {batch_size}")
            return Batch(x, device=device)

        return Batch.broadcast(x, batch_size, device=device)


_unsupported_args = {"bytes_per_sample_hint", "preserve"}


def _find_or_create_module(root_module, module_path):
    module = root_module
    for path_part in module_path:
        submodule = getattr(module, path_part, None)
        if submodule is None:
            submodule = types.ModuleType(path_part)
            setattr(module, path_part, submodule)
        module = submodule
    return module


def build_operator_class(schema):
    class_name = schema.OperatorName()
    module_path = schema.ModulePath()
    is_reader = "readers" in module_path
    if is_reader:
        from .. import dali2 as parent

        module = parent
    else:
        module = ops
    legacy_op_class = None
    import nvidia.dali.ops

    legacy_op_module = nvidia.dali.ops
    for path_part in module_path:
        legacy_op_module = getattr(legacy_op_module, path_part)
    module = _find_or_create_module(module, module_path)

    legacy_op_class = getattr(legacy_op_module, class_name)
    base = ops.Operator
    if "readers" in module.__name__:
        base = ops.Reader
    op_class = type(class_name, (base,), {})
    op_class.schema = schema
    op_class.supported_backends = set(schema.GetSupportedBackends())
    op_class.op_name = class_name
    op_class.fn_name = _to_snake_case(class_name)
    op_class.legacy_op = legacy_op_class
    op_class.is_stateful = schema.IsStateful()
    op_class._instance_cache = {}  # TODO(michalz): Make it thread-local
    op_class.__init__ = build_constructor(schema, op_class)
    op_class.__call__ = build_call_function(schema, op_class)
    op_class.__module__ = module.__name__
    op_class.__qualname__ = class_name
    setattr(module, class_name, op_class)
    return op_class


def build_constructor(schema, op_class):
    stateful = op_class.is_stateful
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
    header_args = [
        "self",
        "max_batch_size=None",
        "name=None",
        'device="cpu"',
        "num_inputs=None",
    ] + call_args
    header = f"__init__({', '.join(header_args)})"

    def init(self, max_batch_size, name, **kwargs):
        kwargs = {k: _scalar_decay(v) for k, v in kwargs.items()}
        op_class.__base__.__init__(self, max_batch_size, name, **kwargs)
        if stateful:
            self._call_id = 0

    function = makefun.create_function(header, init)
    function.__qualname__ = f"{op_class.__name__}.{function_name}"

    return function


def build_call_function(schema, op_class):
    stateful = op_class.is_stateful
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

    call_args = ["*", "batch_size=None"] + call_args
    if inputs:
        inputs = inputs + ["/"]
    header = f"__call__({', '.join(['self'] + inputs + call_args)})"

    def call(self, *raw_args, batch_size=None, **raw_kwargs):
        with nvtx.annotate(f"__call__: {self.op_name}", domain="op_builder"):
            self._pre_call(*raw_args, **raw_kwargs)
            with nvtx.annotate("__call__: get batch size", domain="op_builder"):
                is_batch = batch_size is not None
                if batch_size is None:
                    for i, x in enumerate(list(raw_args) + list(raw_kwargs.values())):
                        x_batch_size = _get_batch_size(x)
                        if x_batch_size is not None:
                            is_batch = True
                            if batch_size is not None:
                                if x_batch_size != batch_size:
                                    raise ValueError(
                                        f"Inconsistent batch size: {x_batch_size} != {batch_size}"
                                    )
                            else:
                                batch_size = x_batch_size
                if not is_batch:
                    batch_size = self._max_batch_size or 1

            inputs = []
            kwargs = {}

            if is_batch:
                with nvtx.annotate("__call__: convert to batches", domain="op_builder"):
                    for i, inp in enumerate(raw_args):
                        if inp is None:
                            continue
                        input_device = self.input_device(i, _get_input_device_type(inp))
                        inp = _to_batch(inp, batch_size, device=input_device)
                        inputs.append(inp)
                    for k, v in raw_kwargs.items():
                        if v is None:
                            continue
                        kwargs[k] = _to_batch(v, batch_size, device=_device.Device("cpu"))
            else:
                with nvtx.annotate("__call__: convert to tensors", domain="op_builder"):
                    for inp in raw_args:
                        if inp is None:
                            continue
                        inputs.append(_to_tensor(inp))
                    for k, v in raw_kwargs.items():
                        if v is None:
                            continue
                        kwargs[k] = _to_tensor(v)

            with nvtx.annotate("__call__: shallowcopy", domain="op_builder"):
                inputs = [copy.copy(x) for x in inputs]
                kwargs = {k: copy.copy(v) for k, v in kwargs.items()}

            if stateful:
                call_id = self._call_id
                self._call_id += 1
            else:
                call_id = None
            with nvtx.annotate("__call__: construct Invocation", domain="op_builder"):
                invocation = _invocation.Invocation(
                    self,
                    call_id,
                    inputs,
                    kwargs,
                    is_batch=is_batch,
                    batch_size=batch_size,
                    previous_invocation=self._last_invocation,
                )

            if stateful:
                self._last_invocation = invocation

            if (
                _eval_mode.EvalMode.current() == _eval_mode.EvalMode.eager
                or _eval_mode.EvalMode.current() == _eval_mode.EvalMode.sync_cpu
                or _eval_mode.EvalMode.current() == _eval_mode.EvalMode.sync_full
                or (
                    _eval_mode.EvalMode.current() == _eval_mode.EvalMode.default
                    and (
                        any(is_external(x) for x in inputs)
                        or any(is_external(x) for x in kwargs.values())
                    )
                )
            ):
                # Evaluate immediately
                invocation.run(_eval_context.EvalContext.get())
            else:
                pass
                # Lazy evaluation
                # If there's an active evaluation context, add this invocation to it.
                # When leaving the context, the invocation will be evaluated if it's still alive.
                # ctx = _eval_context.EvalContext.current()
                # if ctx is not None:
                # ctx._add_invocation(invocation, weak=not self.is_stateful)

            if is_batch:
                if len(invocation) == 1:
                    return Batch(invocation_result=invocation[0])
                else:
                    return tuple(
                        Batch(invocation_result=invocation[i]) for i in range(len(invocation))
                    )
            else:
                if len(invocation) == 1:
                    return Tensor(invocation_result=invocation[0])
                else:
                    return tuple(
                        Tensor(invocation_result=invocation[i]) for i in range(len(invocation))
                    )

    function = makefun.create_function(header, call)

    return function


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def build_fn_wrapper(op):
    schema = op.schema
    module_path = schema.ModulePath()
    from .. import dali2 as parent

    module = parent
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
    signature_args = ["batch_size=None, device=None"]
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

    def fn_call(*inputs, batch_size=None, device=None, **raw_kwargs):
        if batch_size is None:
            for x in inputs:
                x_batch_size = _get_batch_size(x)
                if x_batch_size is not None:
                    batch_size = x_batch_size
                    break
        if batch_size is None:
            for arg in raw_kwargs.values():
                x_batch_size = _get_batch_size(arg)
                if x_batch_size is not None:
                    batch_size = x_batch_size
                    break
        max_batch_size = _next_pow2(batch_size or 1)
        init_args = {
            arg: _scalar_decay(raw_kwargs[arg])
            for arg in fixed_args
            if arg != "max_batch_size" and arg in raw_kwargs and raw_kwargs[arg] is not None
        }
        call_args = {
            arg: _scalar_decay(raw_kwargs[arg])
            for arg in tensor_args
            if arg in raw_kwargs and raw_kwargs[arg] is not None
        }
        # If device is not specified, infer it from the inputs and call_args
        if device is None:

            def _infer_device():
                for inp in inputs:
                    if inp is None:
                        continue
                    dev = _get_input_device(inp)
                    if dev is not None and dev.device_type == "gpu":
                        return dev
                for arg in raw_kwargs.values():
                    if arg is None:
                        continue
                    dev = _get_input_device(arg)
                    if dev is not None and dev.device_type == "gpu":
                        return dev
                return _device.Device("cpu")

            device = _infer_device()
            device_inferred = True
        elif not isinstance(device, _device.Device):
            device = _device.Device(device)
            device_inferred = False

        supported_backends = op.supported_backends
        if device.device_type not in supported_backends:
            if len(supported_backends) == 1 and device_inferred:
                # Maybe we got it wrong? Try the only device that's there
                device.device_type = supported_backends[0]
            else:
                # No we want to call "mixed" operators "gpu" - but we still have distinct backends.
                # Hardly any op has both "mixed" and "gpu", so we can just replace "gpu" with
                # "mixed".
                if device.device_type == "gpu" and "mixed" in supported_backends:
                    device.device_type = "mixed"

        # Get or create the operator instance that matches the arguments
        with nvtx.annotate(f"get instance {op.op_name}", domain="op_builder"):
            op_inst = op.get(
                max_batch_size=max_batch_size,
                name=None,
                device=device,
                num_inputs=len(inputs),
                call_arg_names=tuple(call_args.keys()),
                **init_args,
            )

        # Call the operator (the result is an Invocation object)
        return op_inst(*inputs, batch_size=batch_size, **call_args)

    function = makefun.create_function(header, fn_call)
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
    for what, in_favor in deprecated.items():
        schema = _b.GetSchema(what)
        module = _find_or_create_module(ops, schema.ModulePath())
        setattr(module, what, op_map[in_favor])

    return all_op_classes
