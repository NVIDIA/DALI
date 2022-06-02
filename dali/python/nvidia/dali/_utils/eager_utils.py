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

from nvidia.dali import tensors as _tensors
from nvidia.dali import types as _types
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
    """Classification of data's device and whether it is a batch.

    Based on data type determines if data should be treated as a batch and with which device.
    If the type can be recognized as a batch without being falsely categorized as such, it is.
    This includes lists of supported tensor-like objects e.g. numpy arrays (the only list not
    treated as a batch is a list of objects of primitive types), :class:`DataNodeDebug` and
    TensorLists.

    Args:
        data: Data to be classified.
        type_name (str): Representation of argument type (input or keyword).
        arg_constant_len (int): Only applicable for argument inputs that are of array type
            (e.g. numpy array). If -1 does not modify the data. For positive value works like
            `:class:ops.Constant`, repeats the data `arg_constant_len` times.
    """

    def __init__(self, data, type_name, arg_constant_len=-1):
        self.is_batch, self.device, self.data = self._classify_data(
            data, type_name, arg_constant_len)

    @staticmethod
    def _classify_data(data, type_name, arg_constant_len):
        from nvidia.dali._debug_mode import DataNodeDebug
        """Returns tuple (is_batch, device, unpacked data). """

        def is_primitive_type(x):
            return isinstance(x, (int, float, bool, str))

        def classify_array_input(arr):
            if _types._is_numpy_array(arr):
                device = 'cpu'
            elif _types._is_torch_tensor(arr):
                device = 'gpu' if arr.is_cuda else 'cpu'
            elif _types._is_mxnet_array(arr):
                device = 'gpu' if 'gpu' in str(arr.context) else 'cpu'
            else:
                raise RuntimeError(f"Unsupported array type '{type(arr)}'.")

            return False, device, arr

        def classify_array_kwarg(arr):
            if _types._is_torch_tensor(arr):
                if arr.is_cuda:
                    arr = arr.cpu().numpy()
            elif _types._is_mxnet_array(arr):
                import mxnet as mx

                if 'gpu' in str(arr.context):
                    arr = arr.copyto(mx.cpu())
            elif not _types._is_numpy_array(arr):
                raise RuntimeError(f"Unsupported array type '{type(arr)}'.")

            arr = _types._preprocess_constant_array_type(arr)
            arr = _tensors.TensorListCPU([_tensors.TensorCPU(arr)] * arg_constant_len)
            return True, 'cpu', arr

        if isinstance(data, list):
            if len(data) == 0 or any([is_primitive_type(d) for d in data]):
                return False, 'cpu', data

            is_batch_list = []
            device_list = []
            data_list = []

            for d in data:
                is_batch, device, val = _Classification._classify_data(d, type_name, -1)
                is_batch_list.append(is_batch)
                device_list.append(device)
                data_list.append(val)

            if any([device != device_list[0] for device in device_list]):
                raise RuntimeError(f'{type_name} has batches of data on CPU and on GPU, '
                                   'which is not supported.')

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
            if _types._is_compatible_array_type(data):
                if arg_constant_len > 0:
                    # For call argument input repeats data `arg_constant_len` times to match
                    # the ContantOp behavior.
                    return classify_array_kwarg(data)
                else:
                    return classify_array_input(data)
            if hasattr(data, '__cuda_array_interface__') or isinstance(data, _tensors.TensorGPU):
                return False, 'gpu', data

        return False, 'cpu', data


def _slice_tensorlist(data, size):
    """ Constructs TensorList consisting of ``size`` first elements of ``data``. """

    return type(data)(list(data)[:size], layout=data.layout())
