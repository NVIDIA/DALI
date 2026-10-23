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


import functools
import numbers

from ._call_site import mark_transparent, resolve_callsite_frame


@functools.cache
def _arithmetic_dunders():
    binary_ops = (
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "lshift",
        "rshift",
        "and",
        "or",
        "xor",
        "matmul",
        "divmod",
    )
    # Comparisons have no __r*__ variants
    comparisons = ("eq", "ne", "lt", "le", "gt", "ge")
    unary_ops = ("neg", "pos", "abs", "invert")

    binary_dunders = (f"__{prefix}{stem}__" for stem in binary_ops for prefix in ("", "r"))
    other_dunders = (f"__{stem}__" for stem in (*comparisons, *unary_ops))

    return (*binary_dunders, *other_dunders)


def transparent_arithmetic(cls: type) -> type:
    """Annotate a class' arithmetic dunders with ``mark_transparent``"""
    for dunder in _arithmetic_dunders():
        if func := vars(cls).get(dunder):
            mark_transparent(func)
    return cls


@mark_transparent
def _arithm_op(name: str, *args):
    return _arithmetic_call(name)(*args)


@mark_transparent
def _eager_arithmetic(name, *args, batch_size=None, device=None, _backend=None, _caller_frame=None):
    from . import _arithmetic_generic_op
    from ._batch import Batch
    from ._tensor import Tensor, as_tensor

    tensor_args = [arg for arg in args if isinstance(arg, (Tensor, Batch))]
    gpu = any(arg.device.device_type == "gpu" for arg in tensor_args)

    def to_input(arg):
        if not isinstance(arg, (Tensor, Batch)):
            device = "gpu" if gpu and isinstance(arg, (numbers.Real, list, tuple)) else None
            arg = as_tensor(arg, device=device)

        if (arg.device.device_type == "gpu") != gpu:
            raise ValueError("Cannot mix GPU and CPU inputs.")

        return arg

    if any(type(arg) in (bool, int, float) for arg in args):
        from ._source_analysis import constant_inputs

        constants = constant_inputs(resolve_callsite_frame(depth_hint=3), args)
        if not tensor_args and all(constants):
            constants = (False, *constants[1:])
    else:
        constants = (False,) * len(args)

    desc, inputs, integers, reals = [], [], [], []
    for arg, constant in zip(args, constants, strict=True):
        type_ = type(arg)
        if type_ is int and (arg >> 31) not in (0, -1):
            raise OverflowError(f"Integer {arg} is out of range for int32.")

        if not constant:
            desc.append(f"&{len(inputs)}")
            inputs.append(to_input(arg))
        elif type_ is bool:
            desc.append(f"${len(integers)}:bool")
            integers.append(int(arg))
        elif type_ is int:
            desc.append(f"${len(integers)}:int32")
            integers.append(arg)
        elif type_ is float:
            desc.append(f"${len(reals)}:float32")
            reals.append(arg)
        else:
            desc.append(f"&{len(inputs)}")
            inputs.append(to_input(arg))

    return _arithmetic_generic_op._eager_call(
        *inputs,
        batch_size=batch_size,
        device=device,
        _backend=_backend,
        _caller_frame=_caller_frame,
        expression_desc=f"{name}({' '.join(desc)})",
        integer_constants=integers or None,
        real_constants=reals or None,
    )


@functools.cache
def _arithmetic_call(name: str):
    from . import _arithmetic_generic_op
    from ._capture import _capture_intercept
    from ._ops import Operator

    class Arithmetic(Operator):
        """Capture the original operands, before eager lowering converts them to tensors."""

        _supported_backends = _arithmetic_generic_op._op_class._supported_backends
        _argument_conversion_map = {}

        @classmethod
        def _legacy_op_kwargs(cls):
            return {}  # no schema, and _legacy_op below takes no metadata

        @staticmethod
        def _legacy_op(*, device):
            from nvidia.dali.data_node import DataNode
            from nvidia.dali.ops._operators.math import _arithm_op as pipeline_arithmetic
            from nvidia.dali import fn
            from nvidia.dali.types import Constant, DALIDataType
            from ._tensor import as_tensor
            import numpy as np

            def call(*args):
                inputs = []
                for arg in args:
                    if not isinstance(arg, DataNode) and type(arg) not in (bool, int, float):
                        # Use dynamic mode's dtype inference for tensor constants.
                        tensor = as_tensor(arg)
                        array = np.asarray(tensor.cpu())
                        if array.dtype == np.float64:
                            # fn.constant doesn't support float64, so pass the bits as int64
                            arg = fn.reinterpret(
                                Constant(
                                    array.view(np.int64),
                                    dtype=DALIDataType.INT64,
                                    device=device,
                                ),
                                dtype=DALIDataType.FLOAT64,
                            )
                        else:
                            arg = Constant(array, dtype=tensor.dtype.type_id, device=device)
                    inputs.append(arg)
                return pipeline_arithmetic(name, *inputs)

            return call

    eager = functools.partial(_eager_arithmetic, name)
    return _capture_intercept(eager, Arithmetic, "arithmetic")
