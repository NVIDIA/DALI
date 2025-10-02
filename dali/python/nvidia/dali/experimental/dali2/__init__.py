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

"""
DALI2 is a new experimental API that is currently under development.
"""

from ._eval_mode import *  # noqa: F401, F403
from ._eval_context import *  # noqa: F401, F403
from ._type import *  # noqa: F401, F403
from ._device import *  # noqa: F401, F403
from ._tensor import Tensor  # noqa: F401
from ._batch import Batch  # noqa: F401

# REVIEW ONLY
def _convert_tensor_cpu(tensor, dtype):
    converted = np.array(tensor, dtype=_types.to_numpy_type(dtype.type_id))
    converted_backend = _b.TensorCPU(converted, tensor.GetLayout())
    return converted

# REVIEW ONLY
def cast(tensor_or_batch, dtype, device):
    from . import _type
    import nvidia.dali.backend as _b
    import nvidia.dali.types as _types
    dtype = _type.dtype(dtype)
    if device is None:
        device = tensor_or_batch.device
    cpu = tensor_or_batch.cpu()
    cpu.evaluate()
    if isinstance(cpu, Tensor):
        converted_cpu = Tensor(_convert_tensor_cpu(cpu._backend, dtype))
    else:
        assert(isinstance(cpu, Batch))
        tl = cpu._backend
        converted_cpu = Batch([
            _convert_tensor_cpu(tl[i] for i in range(len(tl)))
        ])
    return converted_cpu.to_device(device)

# REVIEW ONLY
def copy(tenosr_or_batch, device):
    tensor_or_batch.evaluate()
    b = tensor_or_batch._backend
    if device.device_type == "cpu":
        if isisnstance(b, _b.TensorCPU):
            return _b.TensorCPU(b)
        elif isinstance(b, _b.TensorListCPU):
            return _b.TensorListCPU(b)
        else:
            return _b.as_cpu()
    else:
        assert device.device_type == "gpu"
        if isinstance(b, _b.TensorGPU):
            return _b.TensorGPU(
