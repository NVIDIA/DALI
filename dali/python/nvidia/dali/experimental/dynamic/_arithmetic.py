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


import numbers


def _arithm_op(name: str, *args):
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

    # only reachable from math functions called with only scalars, e.g. ndd.math.max(2, 3)
    if not tensor_args and args:
        args = (to_input(args[0]), *args[1:])

    desc, inputs, integers, reals = [], [], [], []
    for arg in args:
        type_ = type(arg)
        if type_ is bool:
            desc.append(f"${len(integers)}:bool")
            integers.append(int(arg))
        elif type_ is int:
            if (arg >> 31) not in (0, -1):
                raise OverflowError(f"Integer constant {arg} is out of range for int32.")
            desc.append(f"${len(integers)}:int32")
            integers.append(arg)
        elif type_ is float:
            desc.append(f"${len(reals)}:float32")
            reals.append(arg)
        else:
            desc.append(f"&{len(inputs)}")
            inputs.append(to_input(arg))

    return _arithmetic_generic_op(
        *inputs,
        expression_desc=f"{name}({' '.join(desc)})",
        integer_constants=integers or None,
        real_constants=reals or None,
    )
