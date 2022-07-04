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

# pylint: disable=no-name-in-module, unused-import
from nvidia.dali.backend import TensorCPU, TensorListCPU, TensorGPU, TensorListGPU      # noqa: F401


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
    data = _transfer_to_cpu(self, type_name[-3:])
    data_str = '[]'
    crop = False

    if data:
        if data.is_dense_tensor():
            data_str = np.array2string(np.array(data.as_tensor()),
                                       prefix=spaces_indent, edgeitems=edgeitems)
        else:
            data = list(map(np.array, data))

            # Triggers summarization if total number of elements exceeds 1000
            # (empty tensor is treated as 1 element).
            crop = len(data) > 2 * edgeitems + 1 and sum(max(arr.size, 1) for arr in data) > 1000
            if crop:
                data = data[:edgeitems] + data[-edgeitems:]

            # Separator matching numpy standard.
            sep = '\n' * data[0].ndim + spaces_indent

            data = [np.array2string(tensor, prefix=spaces_indent, edgeitems=edgeitems)
                    for tensor in data]
            data_str = f'[{_join_string(data, crop, edgeitems, sep)}]'

    shape = self.shape()
    shape_len = len(shape)
    shape_prefix = 'shape=['
    shape_crop = shape_len > 16 or (shape_len > 2 * edgeitems + 1 and
                                    shape_len * len(shape[0]) > 100)
    shape = list(map(str, shape))
    shape_str = _join_string(shape, shape_crop, edgeitems)

    if len(shape_str) > 75:
        # Break shapes into separate lines.
        shape_str = _join_string(shape, shape_crop, edgeitems, ', \n' +
                                 spaces_indent + ' ' * len(shape_prefix))

    params = [f'{type_name}(\n{spaces_indent}{data_str}', f'dtype={self.dtype}'] + \
        ([f'layout="{layout}"'] if layout else []) + \
        [f'num_samples={len(self)}', f'{shape_prefix}{shape_str}])']

    return _join_string(params, False, 0, ',\n' + spaces_indent)
