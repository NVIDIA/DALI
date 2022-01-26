# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pylint: disable=no-name-in-module, unused-import
import nvidia.dali.backend
from nvidia.dali.backend import TensorCPU
from nvidia.dali.backend import TensorListCPU
from nvidia.dali.backend import TensorGPU
from nvidia.dali.backend import TensorListGPU


def _transfer_to_cpu(data, device):
    if device.lower() == 'gpu':
        return data.as_cpu()
    return data


def _join_string(data, crop, edgeitems, sep=', '):
    if crop:
        data = data[:edgeitems] + ['...'] + data[-edgeitems:]

    return sep.join(data)


def _tensor_to_string(self):
    """ Returns string representation of Tensor."""
    import numpy as np

    type_name = type(self).__name__
    indent = ' ' * 4
    layout = self.layout()
    data = np.array(_transfer_to_cpu(self, type_name[-3:]))
    data_str = np.array2string(data, prefix=indent, edgeitems=2)

    params = [f'{type_name}(\n{indent}{data_str}', f'dtype={self.dtype}'] + \
        ([f'layout={layout}'] if layout else []) + \
        [f'shape={self.shape()})']

    return _join_string(params, False, 0, ',\n' + indent)


def _tensorlist_to_string(self, indent=''):
    """ Returns string representation of TensorList."""
    import numpy as np

    edgeitems = 2
    spaces_indent = indent + ' ' * 4
    type_name = type(self).__name__
    layout = self.layout()
    shape = list(map(str, self.shape()))
    data = _transfer_to_cpu(self, type_name[-3:])
    data_str = '[]'
    crop = False

    if data:
        if data.is_dense_tensor():
            data_str = np.array2string(np.array(data.as_tensor()),
                                       prefix=spaces_indent, edgeitems=edgeitems)
        else:
            data = list(data)

            # If number of tensors > 5 returns 2 first and 2 last tensors seperated by dots.
            crop = len(data) > edgeitems * 2 + 1
            if crop:
                data = data[:edgeitems] + data[-edgeitems:]

            data = list(map(np.array, data))

            # Seperator matching numpy standard.
            sep = '\n' * data[0].ndim + spaces_indent

            data = [np.array2string(tensor, prefix=spaces_indent, edgeitems=edgeitems)
                    for tensor in data]
            data_str = f'[{_join_string(data, crop, edgeitems, sep)}]'

    params = [f'{type_name}(\n{spaces_indent}{data_str}', f'dtype={self.dtype}'] + \
        ([f'layout="{layout}"'] if layout else []) + \
        [f'num_samples={len(self)}',
         f'shape=[{_join_string(shape, len(shape) > edgeitems * 2 + 1, edgeitems)}])']

    return _join_string(params, False, 0, ',\n' + spaces_indent)


setattr(TensorCPU, '__str__', _tensor_to_string)
setattr(TensorGPU, '__str__', _tensor_to_string)
setattr(TensorListCPU, '__str__', _tensorlist_to_string)
setattr(TensorListGPU, '__str__', _tensorlist_to_string)

setattr(TensorCPU, '__repr__', TensorCPU.__str__)
setattr(TensorGPU, '__repr__', TensorGPU.__str__)
setattr(TensorListCPU, '__repr__', TensorListCPU.__str__)
setattr(TensorListGPU, '__repr__', TensorListGPU.__str__)
