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
from ._batch import Batch, _get_batch_size, as_batch as _as_batch
from ._tensor import Tensor
from . import ops
from . import _type
import copy
from . import _invocation, _device, _eval_mode, _eval_context
import nvidia.dali.ops as _ops
import nvidia.dali.types
import nvtx
from nvidia.dali import internal as _internal
from nvidia.dali.ops import _docs, _names
from . import random as _random


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


def _to_tensor(x, device=None, dtype=None):
    with nvtx.annotate("to_tensor", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Tensor):
            if dtype is not None and x.dtype != dtype:
                return Tensor(x, dtype=dtype, device=device)
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                raise ValueError("Batch invocation result cannot be used as a single tensor")
            return Tensor(invocation_result=x, device=device)
        return Tensor(x, device=device, dtype=dtype)


def _to_batch(x, batch_size, device=None, dtype=None):
    with nvtx.annotate("to_batch", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Batch):
            if dtype is not None and x.dtype != dtype:
                return _as_batch(x, dtype=dtype, device=device)
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                return Batch(invocation_result=x, device=device, dtype=dtype)
            else:
                x = _to_tensor(x, dtype=dtype)  # fall back to regular replication
        actual_batch_size = _get_batch_size(x)
        if actual_batch_size is not None:
            if batch_size is not None and actual_batch_size != batch_size:
                raise ValueError(f"Unexpected batch size: {actual_batch_size} != {batch_size}")
            return Batch(x, device=device, dtype=dtype)

        return Batch.broadcast(x, batch_size, device=device, dtype=dtype)


_unsupported_args = {"bytes_per_sample_hint", "preserve"}


def _find_or_create_module(root_module, module_path):
    return _internal.get_submodule(root_module, module_path)


def _scalar_arg_type_id(dtype_id):
    if dtype_id == nvidia.dali.types.DALIDataType._INT32_VEC:
        return nvidia.dali.types.INT32
    elif dtype_id == nvidia.dali.types.DALIDataType._FLOAT_VEC:
        return nvidia.dali.types.FLOAT
    elif dtype_id == nvidia.dali.types.DALIDataType._STRING_VEC:
        return nvidia.dali.types.STRING
    elif dtype_id == nvidia.dali.types.DALIDataType._BOOL_VEC:
        return nvidia.dali.types.BOOL
    else:
        return dtype_id


def _argument_type_conversion(dtype_id):
    try:
        return _type.dtype(_scalar_arg_type_id(dtype_id))
    except KeyError:
        return None


def build_operator_class(schema):
    class_name = schema.OperatorName()
    module_path = schema.ModulePath()
    is_reader = "readers" in module_path
    if is_reader:
        from .. import dynamic as parent

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
    op_class.has_random_state_arg = schema.HasRandomStateArg()
    op_class._instance_cache = {}  # TODO(michalz): Make it thread-local
    op_class._generated = True
    op_class.__init__ = build_constructor(schema, op_class)
    op_class.__call__ = build_call_function(schema, op_class)
    op_class.__module__ = module.__name__
    op_class.__qualname__ = class_name
    op_class._argument_conversion_map = {
        arg: _argument_type_conversion(schema.GetArgumentType(arg))
        for arg in schema.GetArgumentNames(include_hidden=True)
    }
    setattr(module, class_name, op_class)
    return op_class


def build_constructor(schema, op_class):
    stateful = op_class.is_stateful
    function_name = "__init__"

    init_args = []
    used_kwargs = set()
    for arg in schema.GetArgumentNames():
        if arg in _unsupported_args:
            continue
        if schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
            init_args.append(f"{arg}=None")
        else:
            init_args.append(arg)
        used_kwargs.add(arg)

    if init_args:
        init_args = ["*"] + init_args
    header_args = [
        "self",
        "max_batch_size=None",
        "name=None",
        'device="cpu"',
        "num_inputs=None",
    ] + init_args
    header = f"__init__({', '.join(header_args)})"

    def init(self, max_batch_size, name, **kwargs):
        kwargs = {k: _scalar_decay(v) for k, v in kwargs.items()}
        op_class.__base__.__init__(self, max_batch_size, name, **kwargs)
        if stateful:
            self._call_id = 0

    doc = _docs._docstring_generator_class(schema.Name(), api="dynamic", args=used_kwargs)
    function = makefun.create_function(header, init, doc=doc)
    function.__qualname__ = f"{op_class.__name__}.{function_name}"

    return function


def _get_inputs(schema):
    inputs = []
    min_inputs = schema.MinNumInput()
    max_inputs = schema.MaxNumInput()
    num_separate_inputs = min_inputs
    if schema.HasInputDox() or max_inputs <= _docs._MAX_INPUT_SPELLED_OUT:
        num_separate_inputs = max_inputs

    for i in range(num_separate_inputs):
        name = _names._get_input_name(schema, i)
        if i < min_inputs:
            inputs.append(name)
        else:
            inputs.append(f"{name}=None")

    if inputs:
        inputs.append("/")
    if num_separate_inputs < max_inputs:
        inputs.append("*inputs")
    else:
        inputs.append("*")
    return inputs


def build_call_function(schema, op_class):
    stateful = op_class.is_stateful
    has_random_state_arg = op_class.has_random_state_arg
    call_args = []
    used_kwargs = set()
    for arg in schema.GetArgumentNames():
        if arg in _unsupported_args:
            continue
        if not schema.IsTensorArgument(arg):
            continue
        if schema.IsArgumentOptional(arg):
            call_args.append(f"{arg}=None")
        else:
            call_args.append(arg)
        used_kwargs.add(arg)

    call_args = ["batch_size=None"] + call_args

    # Add rng argument for random operators
    if has_random_state_arg:
        call_args.append("rng=None")
        used_kwargs.add("rng")
        # Remove 'seed' from used_kwargs and signature_args if present
        if "seed" in used_kwargs:
            used_kwargs.remove("seed")

    inputs = _get_inputs(schema)

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

            if has_random_state_arg:
                rng = raw_kwargs.pop("rng", None)
                if rng is None:
                    rng = _random.get_default_rng()
                if not isinstance(rng, _random.RNG):
                    raise ValueError(
                        f"rng must be an instance of nvidia.dali.experimental.dynamic.random.RNG, "
                        f"but got {type(rng)}"
                    )

                # Use the provided RNG to generate 7 random uint32 values.
                # This creates a fixed-size random state tensor.
                # 7 uint32 words = 224 bits; required is 194 bits (operator reads first 25 bytes).
                # Only one random state tensor is created per call, not per sample.
                raw_kwargs["_random_state"] = Tensor(
                    [rng() for _ in range(7)],
                    dtype=_type.dtype(nvidia.dali.types.UINT32),
                    device="cpu",
                )

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
                        dtype = op_class._argument_conversion_map[k]
                        kwargs[k] = _to_batch(
                            v, batch_size, device=_device.Device("cpu"), dtype=dtype
                        )
            else:
                with nvtx.annotate("__call__: convert to tensors", domain="op_builder"):
                    for inp in raw_args:
                        if inp is None:
                            continue
                        inputs.append(_to_tensor(inp))
                    for k, v in raw_kwargs.items():
                        if v is None:
                            continue
                        dtype = op_class._argument_conversion_map[k]
                        kwargs[k] = _to_tensor(v, dtype=dtype)

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
                invocation.run(_eval_context.EvalContext.current())
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

    doc = _docs._docstring_generator_call(schema.Name(), api="dynamic", args=used_kwargs)
    function = makefun.create_function(header, call, doc=doc)

    return function


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def build_fn_wrapper(op):
    schema = op.schema
    module_path = schema.ModulePath()
    from .. import dynamic as parent

    module = _internal.get_submodule(parent, module_path)

    fn_name = _to_snake_case(op.schema.OperatorName())
    inputs = _get_inputs(schema)

    fixed_args = []
    tensor_args = []
    signature_args = ["batch_size=None, device=None"]
    used_kwargs = set()

    for arg in op.schema.GetArgumentNames():
        if arg in _unsupported_args:
            continue
        if op.schema.IsTensorArgument(arg):
            tensor_args.append(arg)
        else:
            fixed_args.append(arg)
        used_kwargs.add(arg)
        if op.schema.IsArgumentOptional(arg):
            signature_args.append(f"{arg}=None")
        else:
            signature_args.append(arg)

    if schema.HasRandomStateArg():
        tensor_args.append("rng")
        used_kwargs.add("rng")
        signature_args.append("rng=None")
        # Remove 'seed' from used_kwargs and signature_args if present
        if "seed" in used_kwargs:
            used_kwargs.remove("seed")
        if "seed" in signature_args:
            signature_args.remove("seed")

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

    doc = _docs._docstring_generator_fn(schema.Name(), api="dynamic", args=used_kwargs)
    function = makefun.create_function(header, fn_call, doc=doc)
    function.op_class = op
    function.schema = schema
    function._generated = True
    function.__module__ = module.__name__
    setattr(module, fn_name, function)
    return function


def build_fn_wrappers(all_ops):
    wrappers = []
    for op in all_ops:
        if op.op_name.startswith("_"):
            continue
        # Allow random operators to have functional wrappers even if stateful
        if op.schema.IsStateful() and not op.schema.HasRandomStateArg():
            continue

        wrappers.append(build_fn_wrapper(op))
    return wrappers


def build_operators():
    _all_ops = _ops._registry._all_registered_ops()
    all_op_classes = []
    deprecated = {}
    op_map = {}
    for op_name in _all_ops:
        if (
            op_name.endswith("ExternalSource")
            or op_name.endswith("PythonFunction")
            or op_name.endswith("NumbaFunction")
            or op_name.endswith("JaxFunction")
        ):
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

    all_fn_wrappers = build_fn_wrappers(all_op_classes)

    return all_op_classes, all_fn_wrappers
