# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.backend import (  # noqa: F401
    TensorCPU as TensorCPU,
    TensorListCPU as TensorListCPU,
    TensorGPU as TensorGPU,
    TensorListGPU as TensorListGPU,
)


def _transfer_to_cpu(data, device):
    if device.lower() == "gpu":
        return data.as_cpu()
    return data


def _join_string(data, crop, edgeitems, sep=", "):
    if crop:
        data = data[:edgeitems] + ["..."] + data[-edgeitems:]

    return sep.join(data)


np = None


def import_numpy():
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "Could not import numpy. Numpy is required for "
                "Tensor and TensorList printing. "
                "Please make sure you have numpy installed."
            )


def _tensor_to_string(self, show_data=True):
    """Returns string representation of Tensor.

    Parameters
    ----------
    show_data : bool, optional
        Access and format the underlying data, by default True
    """
    import_numpy()

    type_name = type(self).__name__
    indent = " " * 4
    layout = self.layout()
    if show_data:
        data = np.array(_transfer_to_cpu(self, type_name[-3:]))
        data_str = np.array2string(data, prefix=indent, edgeitems=2)

    params = (
        ([f"{data_str}"] if show_data else [])
        + [f"dtype={self.dtype}"]
        + ([f"layout={layout}"] if layout else [])
        + [f"shape={self.shape()})"]
    )

    return f"{type_name}(\n{indent}" + _join_string(params, False, 0, ",\n" + indent)


def _tensorlist_to_string(self, show_data=True, indent=""):
    """Returns string representation of TensorList.

    Parameters
    ----------
    show_data : bool, optional
        Access and format the underlying data, by default True
    indent : str, optional
        optional indentation used in formatting, by default ""
    """
    import_numpy()

    edgeitems = 2
    spaces_indent = indent + " " * 4
    type_name = type(self).__name__
    layout = self.layout()
    if show_data:
        data = _transfer_to_cpu(self, type_name[-3:])
        data_str = "[]"
    else:
        data = None
        data_str = ""

    crop = False

    if data:
        if data.is_dense_tensor():
            data_str = np.array2string(
                np.array(data.as_tensor()), prefix=spaces_indent, edgeitems=edgeitems
            )
        else:
            data = list(map(np.array, data))

            # Triggers summarization if total number of elements exceeds 1000
            # (empty tensor is treated as 1 element).
            crop = len(data) > 2 * edgeitems + 1 and sum(max(arr.size, 1) for arr in data) > 1000
            if crop:
                data = data[:edgeitems] + data[-edgeitems:]

            # Separator matching numpy standard.
            sep = "\n" * data[0].ndim + spaces_indent

            data = [
                np.array2string(tensor, prefix=spaces_indent, edgeitems=edgeitems)
                for tensor in data
            ]
            data_str = f"[{_join_string(data, crop, edgeitems, sep)}]"

    shape = self.shape()
    shape_len = len(shape)
    shape_prefix = "shape=["
    shape_crop = shape_len > 16 or (
        shape_len > 2 * edgeitems + 1 and shape_len * len(shape[0]) > 100
    )
    shape = list(map(str, shape))
    shape_str = _join_string(shape, shape_crop, edgeitems)

    if len(shape_str) > 75:
        # Break shapes into separate lines.
        shape_str = _join_string(
            shape, shape_crop, edgeitems, ", \n" + spaces_indent + " " * len(shape_prefix)
        )

    params = (
        ([f"{data_str}"] if show_data else [])
        + [f"dtype={self.dtype}"]
        + ([f'layout="{layout}"'] if layout else [])
        + [f"num_samples={len(self)}", f"{shape_prefix}{shape_str}])"]
    )

    return f"{type_name}(\n{spaces_indent}" + _join_string(params, False, 0, ",\n" + spaces_indent)
