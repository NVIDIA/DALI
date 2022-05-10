# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.tensors as _tensors
import nvidia.dali.types as _types
from nvidia.dali.external_source import _prep_data_for_feed_input


def _transform_data_to_tensorlist(data, batch_size, layout=None, device_id=None):
    data = _prep_data_for_feed_input(data, batch_size, layout, device_id)

    if isinstance(data, list):
        if isinstance(data[0], _tensors.TensorGPU):
            data = _tensors.TensorListGPU(data, layout or "")
        else:
            data = _tensors.TensorListCPU(data, layout or "")

    return data


class _Classification:
    """Classification of data's device and if it is a batch.

    Based on data type determines if data should be treated as a batch and with which device.
    If the type can be recognized as a batch without being falsely categorized as such, it is.
    This includes lists of supported tensor-like objects e.g. numpy arrays (the only list not
    treated as a batch is a list of objects of primitive types), :class:`DataNodeDebug` and
    TensorLists.
    """

    def __init__(self, data, type_name):
        self.is_batch, self.device, self.data = self._classify_data(data, type_name)

    @staticmethod
    def _classify_data(data, type_name):
        from nvidia.dali._debug_mode import DataNodeDebug
        """Returns tuple (is_batch, device, unpacked data). """

        def is_primitive_type(x):
            return isinstance(x, (int, float, bool, str))

        if isinstance(data, list):
            if len(data) == 0 or any([is_primitive_type(d) for d in data]):
                return False, 'cpu', data

            is_batch_list = []
            device_list = []
            data_list = []

            for d in data:
                is_batch, device, val = _Classification._classify_data(d, type_name)
                is_batch_list.append(is_batch)
                device_list.append(device)
                data_list.append(val)

            if any([device != device_list[0] for device in device_list]):
                raise RuntimeError(f'{type_name} has batches of data on CPU and on GPU. '
                                   'Which is not supported.')

            if all(is_batch_list):
                # Input set.
                return is_batch_list, device_list[0], data_list
            if not any(is_batch_list):
                # Converting to TensorList.
                return True, device_list[0], _transform_data_to_tensorlist(data_list, len(data_list))
            else:
                raise RuntimeError(f'{type_name} has inconsistent batch classification.')

        else:
            if isinstance(data, DataNodeDebug):
                return True, data.device, data.get()
            if isinstance(data, _tensors.TensorListCPU):
                return True, 'cpu', data
            if isinstance(data, _tensors.TensorListGPU):
                return True, 'gpu', data
            if is_primitive_type(data) or isinstance(data, _tensors.TensorCPU):
                return False, 'cpu', data
            if _types._is_numpy_array(data):
                return False, 'cpu', _types._preprocess_constant(data)[0]
            if _types._is_torch_tensor(data):
                device = 'gpu' if data.is_cuda else 'cpu'
                return False, device, _types._preprocess_constant(data)[0]
            if _types._is_mxnet_array(data):
                device = 'gpu' if 'gpu' in str(data.context) else 'cpu'
                return False, device, _types._preprocess_constant(data)[0]
            if hasattr(data, '__cuda_array_interface__') or isinstance(data, _tensors.TensorGPU):
                return False, 'gpu', data

        return False, 'cpu', data
