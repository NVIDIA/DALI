# Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Generalized tensor formatting for both Pipeline API (backend types) and Dynamic API.
Provides abstraction layer for formatting tensors and tensorlists with customization points.
"""

from typing import Any, Optional, Protocol, Union

import numpy as np


def _join_string(data, edgeitems=None, sep=", ", force_ellipsis=False):
    """Join strings with optional cropping for summarization.

    Parameters
    ----------
    data : list
        List of strings to join.
    edgeitems : int or None, optional
        If specified, crops the list to show only the first and last `edgeitems` elements
        with "..." in between. If None, shows all elements. Default: None.
    sep : str, optional
        Separator to use when joining strings. Default: ", ".
    force_ellipsis : bool, optional
        If True and edgeitems is not None, forces ellipsis insertion at position edgeitems
        regardless of list length. Useful when data is pre-cropped. Default: False.

    Returns
    -------
    str
        Joined string, optionally with summarization.
    """
    if edgeitems is not None and force_ellipsis:
        # Force ellipsis at edgeitems position for pre-cropped data
        data = data[:edgeitems] + ["..."] + data[edgeitems:]
    elif edgeitems is not None and len(data) > 2 * edgeitems + 1:
        # Crop and insert ellipsis
        data = data[:edgeitems] + ["..."] + data[-edgeitems:]
    return sep.join(data)


class TensorAdapter(Protocol):
    """Protocol for adapting different tensor types to formatting interface."""

    def get_type_name(self, obj: Any) -> str:
        """Returns the type name for display."""
        ...

    def get_device(self, obj: Any) -> str:
        """Returns device string like 'CPU' or 'GPU'."""
        ...

    def get_dtype(self, obj: Any) -> Any:
        """Returns the data type."""
        ...

    def get_shape(self, obj: Any) -> Union[tuple, list]:
        """Returns the shape as a tuple or list."""
        ...

    def get_layout(self, obj: Any) -> Optional[str]:
        """Returns the layout string, or None if not available."""
        ...

    def to_cpu(self, obj: Any) -> Any:
        """Transfers object to CPU if needed, returns CPU version."""
        ...

    def to_numpy(self, obj: Any) -> Any:
        """Converts object to numpy array."""
        ...


class BatchAdapter(TensorAdapter, Protocol):
    """Protocol for batch/tensorlist types - extends TensorAdapter with batch-specific methods."""

    def get_length(self, obj) -> int:
        """Returns the number of samples in the batch."""
        ...

    def get_sample(self, obj, index: int):
        """Returns a single sample at the given index."""
        ...

    def to_numpy(self, obj, edgeitems=None):
        """Converts object to numpy array.

        Parameters
        ----------
        obj
            The batch object to convert.
        edgeitems : int or None, optional
            If specified and the batch has more than 2*edgeitems+1 samples,
            returns only the first and last edgeitems samples (for summarization).
            If None, returns all samples. Default: None.

        Returns
        -------
        np.ndarray or list of np.ndarray
            Converted data, potentially cropped if edgeitems is specified.
        """
        ...


# Pipeline API adapters (TensorCPU/GPU, TensorListCPU/GPU)
class _BasePipelineAdapter:
    """Base adapter for Pipeline API backend types (TensorCPU/GPU, TensorListCPU/GPU).

    Implements common methods for TensorAdapter protocol.
    Pipeline API uses method calls for shape() and layout().
    """

    def get_type_name(self, obj) -> str:
        return type(obj).__name__

    def get_device(self, obj) -> str:
        type_name = type(obj).__name__
        return type_name[-3:]  # "CPU" or "GPU"

    def get_dtype(self, obj):
        return obj.dtype

    def get_shape(self, obj) -> Union[tuple, list]:
        return obj.shape()

    def get_layout(self, obj) -> Optional[str]:
        return obj.layout()

    def to_cpu(self, obj):
        device = self.get_device(obj)
        if device.lower() == "gpu":
            return obj.as_cpu()
        return obj


class PipelineTensorAdapter(_BasePipelineAdapter):
    """Adapter for TensorCPU/TensorGPU. Implements TensorAdapter protocol."""

    def to_numpy(self, obj):
        return np.array(self.to_cpu(obj))


class PipelineBatchAdapter(_BasePipelineAdapter):
    """Adapter for TensorListCPU/TensorListGPU. Implements BatchAdapter protocol."""

    def get_length(self, obj) -> int:
        return len(obj)

    def get_sample(self, obj, index: int):
        return obj[index]

    def to_numpy(self, obj, edgeitems=None):
        cpu_obj = self.to_cpu(obj)
        length = len(cpu_obj)

        if length == 0:
            return []

        # Check if we should crop
        if edgeitems is not None and length > 2 * edgeitems + 1:
            # Only convert edge samples
            indices = list(range(edgeitems)) + list(range(length - edgeitems, length))
            return [np.array(cpu_obj[i]) for i in indices]

        return [np.array(cpu_obj[i]) for i in range(length)]


# Dynamic API adapters (ndd.Tensor, ndd.Batch)
class _BaseDynamicAdapter:
    """Base adapter for Dynamic API types (ndd.Tensor, ndd.Batch).

    Implements common methods for TensorAdapter protocol.
    Dynamic API uses properties (not methods) for shape and layout.
    """

    def get_type_name(self, obj) -> str:
        return type(obj).__name__

    def get_device(self, obj) -> str:
        return obj.device.device_type.upper()

    def get_dtype(self, obj):
        return obj.dtype

    def get_shape(self, obj) -> Union[tuple, list]:
        return obj.shape

    def get_layout(self, obj) -> Optional[str]:
        return obj.layout

    def to_cpu(self, obj):
        return obj.cpu()


class DynamicTensorAdapter(_BaseDynamicAdapter):
    """Adapter for ndd.Tensor. Implements TensorAdapter protocol."""

    def to_numpy(self, obj):
        cpu_obj = self.to_cpu(obj)
        # Use numpy array protocol (public API)
        return np.array(cpu_obj)


class DynamicBatchAdapter(_BaseDynamicAdapter):
    """Adapter for ndd.Batch. Implements BatchAdapter protocol."""

    def get_length(self, obj) -> int:
        return obj.batch_size

    def get_sample(self, obj, index: int):
        return obj.tensors[index]

    def to_numpy(self, obj, edgeitems=None):
        cpu_obj = self.to_cpu(obj)
        length = cpu_obj.batch_size

        if length == 0:
            return []

        # Determine which samples to convert
        if edgeitems is not None and length > 2 * edgeitems + 1:
            indices = list(range(edgeitems)) + list(range(length - edgeitems, length))
        else:
            indices = range(length)

        return [np.array(cpu_obj.tensors[i]) for i in indices]


def format_tensor(obj, show_data: bool = True, adapter: Optional[TensorAdapter] = None) -> str:
    """Format a single tensor for display.

    Parameters
    ----------
    obj : Any
        The tensor object to format.
    show_data : bool, optional
        Whether to include the actual data values, by default True.
    adapter : TensorAdapter, optional
        Adapter for accessing tensor properties. If None, uses PipelineTensorAdapter.

    Returns
    -------
    str
        Formatted string representation of the tensor.
    """

    if adapter is None:
        adapter = PipelineTensorAdapter()

    indent = " " * 4
    edgeitems = 2
    type_name = adapter.get_type_name(obj)
    layout = adapter.get_layout(obj)
    device = adapter.get_device(obj).lower()

    if show_data:
        data = adapter.to_numpy(obj)
        data_str = np.array2string(data, prefix=indent, edgeitems=edgeitems)
    else:
        data_str = None

    shape = tuple(adapter.get_shape(obj))

    params = (
        ([f"{data_str}"] if show_data else [])
        + [f"dtype={adapter.get_dtype(obj)}"]
        + ([f'layout="{layout}"'] if layout else [])
        + [f'device="{device}"']
        + [f"shape={shape})"]
    )

    return f"{type_name}(\n{indent}" + _join_string(params, sep=",\n" + indent)


def format_batch(
    obj, show_data: bool = True, indent: str = "", adapter: Optional[BatchAdapter] = None
) -> str:
    """Format a tensorlist/batch for display.

    Parameters
    ----------
    obj : Any
        The tensorlist/batch object to format.
    show_data : bool, optional
        Whether to include the actual data values, by default True.
    indent : str, optional
        Optional indentation prefix for the output, by default "".
    adapter : BatchAdapter, optional
        Adapter for accessing batch properties. If None, uses PipelineBatchAdapter.

    Returns
    -------
    str
        Formatted string representation of the tensorlist/batch.
    """

    if adapter is None:
        adapter = PipelineBatchAdapter()

    spaces_indent = indent + " " * 4
    edgeitems = 2
    edgeitem_samples = 2
    type_name = adapter.get_type_name(obj)
    layout = adapter.get_layout(obj)
    device = adapter.get_device(obj).lower()

    if show_data:
        data = adapter.to_cpu(obj)
        data_str = "[]"
    else:
        data = None
        data_str = ""

    crop = False

    if data:
        if adapter.get_length(data) == 0:
            data_str = "[]"
        else:
            # First check if we need to crop based on shapes
            shapes = adapter.get_shape(data)
            num_samples = len(shapes)

            # Compute total elements from shapes to decide if we need summarization
            # (empty tensor is treated as 1 element).
            total_elements = sum(max(np.prod(shape, dtype=int), 1) for shape in shapes)
            crop = num_samples > 2 * edgeitem_samples + 1 and total_elements > 1000

            # Let adapter handle the cropping efficiently
            data_arrays = adapter.to_numpy(data, edgeitems=edgeitem_samples if crop else None)

            # Separator matching numpy standard.
            sep = "\n" * data_arrays[0].ndim + spaces_indent

            # Convert samples to strings
            data_arrays = [
                np.array2string(tensor, prefix=spaces_indent, edgeitems=edgeitems)
                for tensor in data_arrays
            ]
            joined = _join_string(
                data_arrays, edgeitems=edgeitem_samples, sep=sep, force_ellipsis=crop
            )
            data_str = f"[{joined}]"

    shape = adapter.get_shape(obj)
    shape_len = len(shape)
    shape_prefix = "shape=["
    shape_crop = shape_len > 16 or (
        shape_len > 2 * edgeitems + 1 and shape_len * len(shape[0]) > 100
    )
    shape_strs = list(map(str, shape))
    shape_str = _join_string(shape_strs, edgeitems=edgeitems if shape_crop else None)

    if len(shape_str) > 75:
        # Break shapes into separate lines.
        shape_str = _join_string(
            shape_strs,
            edgeitems=edgeitems if shape_crop else None,
            sep=", \n" + spaces_indent + " " * len(shape_prefix),
        )

    params = (
        ([f"{data_str}"] if show_data else [])
        + [f"dtype={adapter.get_dtype(obj)}"]
        + ([f'layout="{layout}"'] if layout else [])
        + [f'device="{device}"']
        + [f"num_samples={adapter.get_length(obj)}", f"{shape_prefix}{shape_str}])"]
    )

    return f"{type_name}(\n{spaces_indent}" + _join_string(params, sep=",\n" + spaces_indent)
