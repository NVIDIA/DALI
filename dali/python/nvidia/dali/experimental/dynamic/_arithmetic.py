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
from typing import Any


def _implicitly_convertible(value: Any):
    return isinstance(value, (numbers.Real, list, tuple))


def _arithm_op(name: str, *args):
    from . import _arithmetic_generic_op
    from ._batch import Batch
    from ._tensor import Tensor, as_tensor

    # scalar arguments are turned into tensors
    argsstr = " ".join(f"&{i}" for i in range(len(args)))
    gpu = any(arg.device.device_type == "gpu" for arg in args if isinstance(arg, (Tensor, Batch)))

    new_args = []
    for arg in args:
        if not isinstance(arg, (Tensor, Batch)):
            if gpu and _implicitly_convertible(arg):
                arg = as_tensor(arg, device="gpu")
            else:
                arg = as_tensor(arg)

        if (arg.device.device_type == "gpu") != gpu:
            raise ValueError("Cannot mix GPU and CPU inputs.")

        new_args.append(arg)

    return _arithmetic_generic_op(*new_args, expression_desc=f"{name}({argsstr})")
