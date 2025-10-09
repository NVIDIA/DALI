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
from ._tensor import *  # noqa: F401, F403
from ._batch import *  # noqa: F401, F403
from ._tensor import Tensor, tensor, as_tensor  # noqa: F401
from ._batch import Batch, batch, as_batch  # noqa: F401


# REVIEW ONLY
def _convert_tensor_cpu(tensor, dtype):
    import numpy as np
    import nvidia.dali.types as _types
    import nvidia.dali.backend as _b

    converted = np.array(tensor, dtype=_types.to_numpy_type(dtype.type_id))
    converted_backend = _b.TensorCPU(converted, tensor.layout())
    return converted_backend


# REVIEW ONLY
def cast(tensor_or_batch, dtype, device=None):
    if device is not None:
        if not isinstance(device, Device):
            device = Device(device)
        assert tensor_or_batch.device == device
    from . import _type

    dtype = _type.dtype(dtype)
    cpu = tensor_or_batch.cpu()
    cpu.evaluate()
    if isinstance(cpu, Tensor):
        converted_cpu = Tensor(_convert_tensor_cpu(cpu._backend, dtype))
    else:
        assert isinstance(cpu, Batch)
        tl = cpu._backend
        converted_cpu = Batch([_convert_tensor_cpu(tl[i], dtype) for i in range(len(tl))])
    return converted_cpu.to_device(tensor_or_batch.device)


# REVIEW ONLY
def copy(tensor_or_batch, device):
    tensor_or_batch.evaluate()
    b = tensor_or_batch._backend
    device_type = device if isinstance(device, str) else device.device_type
    import nvidia.dali.backend as _b

    if device_type == "cpu":
        if isinstance(b, (_b.TensorCPU, _b.TensorListCPU)):
            copied_backend = b._make_copy()
        else:
            copied_backend = b.as_cpu()
    else:
        assert device_type == "gpu"
        if isinstance(b, (_b.TensorGPU, _b.TensorListGPU)):
            copied_backend = b._make_copy()
        else:
            copied_backend = b._as_gpu()
    return type(tensor_or_batch)(copied_backend)


# REVIEW ONLY
def tensor_subscript(target, **kwargs):
    import numpy as np

    ranges = [slice(None)] * target.ndim
    processed = 0
    layout = target.layout or None
    out_layout = ""
    for i in range(len(ranges)):
        if f"at_{i}" in kwargs:
            assert f"lo_{i}" not in kwargs and f"hi_{i}" not in kwargs and f"step_{i}" not in kwargs
            ranges[i] = kwargs[f"at_{i}"]
            processed += 1
        else:
            assert f"at_{i}" not in kwargs
            lo = kwargs.get(f"lo_{i}", None)
            hi = kwargs.get(f"hi_{i}", None)
            step = kwargs.get(f"step_{i}", None)
            ranges[i] = slice(lo, hi, step)
            if lo is not None:
                processed += 1
            if hi is not None:
                processed += 1
            if step is not None:
                processed += 1
            if layout is not None:
                out_layout += layout[i]
    out_layout = out_layout or None

    assert processed == len(kwargs)

    ranges = tuple(ranges)

    def idx_sample(s, idx):
        if s is None:
            return None
        elif isinstance(s, Batch):
            return int(np.array(s.tensors[idx].evaluate()._backend))
        elif isinstance(s, Tensor):
            return int(np.array(s.evaluate()._backend))
        else:
            return s

    def slice_sample(s, idx):
        if isinstance(s, slice):
            start = idx_sample(s.start, idx)
            stop = idx_sample(s.stop, idx)
            step = idx_sample(s.step, idx)
            return slice(start, stop, step)
        else:
            return idx_sample(s, idx)

    def subscript_sample(idx):
        return tuple(slice_sample(s, idx) for s in ranges)

    def do_slice(t, ranges):
        c = t.cpu().evaluate()
        return tensor(np.array(np.array(c._backend)[ranges]), device=t.device, layout=out_layout)

    if isinstance(target, Batch):
        return Batch(
            [do_slice(t, subscript_sample(i)) for i, t in enumerate(target)],
            layout=out_layout,
        )
    else:
        return do_slice(target, ranges)
