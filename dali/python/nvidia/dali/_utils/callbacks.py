# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from nvidia.dali import types
from nvidia.dali import tensors
np = None

def import_numpy():
    """Import numpy lazily, need to define global `np = None` variable"""
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('Could not import numpy. Please make sure you have numpy '
                               'installed before you use parallel mode.')


class _SourceKind(Enum):
    CALLABLE       = 0
    ITERABLE       = 1
    GENERATOR_FUNC = 2

class _SourceDescription:
    """Keep the metadata about the source parameter that was originally passed
    """
    def __init__(self, source, kind: _SourceKind, has_inputs: bool, cycle: str):
        self.source = source
        self.kind = kind
        self.has_inputs = has_inputs
        self.cycle = cycle

    def __str__(self) -> str:
        if self.kind == _SourceKind.CALLABLE:
            return "Callable source " + ("with" if self.has_inputs else "without") + " inputs: `{}`".format(self.source)
        elif self.kind == _SourceKind.ITERABLE:
            return "Iterable (or iterator) source: `{}` with cycle: `{}`.".format(self.source, self.cycle)
        else:
            return "Generator function source: `{}` with cycle: `{}`.".format(self.source, self.cycle)


def _assert_cpu_sample_data_type(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(sample, np.ndarray):
        return True
    if types._is_mxnet_array(sample):
        if sample.ctx.device_type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got an MXNet GPU tensor.")
        return True
    if types._is_torch_tensor(sample):
        if sample.device.type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got a PyTorch GPU tensor.")
        return True
    elif isinstance(sample, tensors.TensorCPU):
        return True
    raise TypeError(error_str.format(type(sample)))


def _assert_cpu_batch_data_type(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    if isinstance(batch, tensors.TensorListCPU):
        return True
    elif isinstance(batch, list):
        for sample in batch:
            _assert_cpu_sample_data_type(sample, error_str)
        return True
    elif _assert_cpu_sample_data_type(batch, error_str):
        # Bach can be repsented as dense tensor
        return True
    else:
        raise TypeError(error_str.format(type(batch)))


def _sample_to_numpy(sample, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    _assert_cpu_sample_data_type(sample, error_str)
    if isinstance(sample, np.ndarray):
        return sample
    if types._is_mxnet_array(sample):
        if sample.ctx.device_type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got an MXNet GPU tensor.")
        return sample.asnumpy()
    if types._is_torch_tensor(sample):
        if sample.device.type != 'cpu':
            raise TypeError("Unsupported callback return type. "
                            "GPU tensors are not supported. Got a PyTorch GPU tensor.")
        return sample.numpy()
    elif isinstance(sample, tensors.TensorCPU):
        return np.array(sample)
    raise TypeError(error_str.format(type(sample)))


def _batch_to_numpy(batch, error_str="Unsupported callback return type. Got: `{}`."):
    import_numpy()
    _assert_cpu_batch_data_type(batch, error_str)
    if isinstance(batch, tensors.TensorListCPU):
        # TODO(klecki): samples that are not uniform
        return batch.as_array()
    elif isinstance(batch, list):
        return [_sample_to_numpy(sample, error_str) for sample in batch]
    else:
        return _sample_to_numpy(batch, error_str)
