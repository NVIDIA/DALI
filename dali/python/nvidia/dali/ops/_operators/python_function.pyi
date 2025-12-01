# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, Optional, overload
from typing import Any, Callable, Sequence, Tuple

from nvidia.dali.data_node import DataNode
from nvidia.dali._typing import TensorLikeIn

class PythonFunction:
    """
    Executes a Python function.

    This operator can be used to execute custom Python code in the DALI pipeline.
    The function receives the data from DALI as NumPy arrays in case of CPU operators or
    as CuPy arrays for GPU operators. It is expected to return the results in the same format. For
    a more universal data format, see :meth:`nvidia.dali.fn.dl_tensor_python_function`.
    The function should not modify input tensors.
    """

    def __init__(
        self,
        /,
        function: Optional[Callable[..., Union[Any, Tuple[Any, ...], None]]] = None,
        num_outputs: int = 1,
        device: str = "cpu",
        batch_processing: bool = False,
        *,
        output_layouts: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        *input: Union[DataNode, TensorLikeIn],
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> Union[DataNode, Sequence[DataNode], None]:
        """See :meth:`nvidia.dali.ops.PythonFunction` class for complete information."""
        ...

class DLTensorPythonFunction:
    """
    Executes a Python function that operates on DLPack tensors.

    The function should not modify input tensors.

    For the GPU operator, it is the user's responsibility to synchronize the device code with DALI.
    To synchronize the device code with DALI, synchronize DALI's work before the operator call
    with the `synchronize_stream` flag (enabled by default) and ensure that the scheduled device
    tasks are finished in the operator call. The GPU code can be executed on the CUDA stream used
    by DALI, which can be obtained by calling the ``current_dali_stream()`` function. In this case,
    the `synchronize_stream` flag can be set to False.
    """

    def __init__(
        self,
        /,
        function: Optional[Callable[..., Union[Any, Tuple[Any, ...], None]]] = None,
        num_outputs: int = 1,
        device: str = "cpu",
        batch_processing: bool = True,
        synchronize_stream: Optional[bool] = True,
        *,
        output_layouts: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        *input: Union[DataNode, TensorLikeIn],
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> None:
        """See :meth:`nvidia.dali.ops.DLTensorPythonFunction` class for complete information."""
        ...

@overload
def python_function(
    *input: Union[DataNode, TensorLikeIn],
    function: Callable[..., Union[Any, Tuple[Any, ...], None]],
    batch_processing: bool = False,
    output_layouts: Union[Sequence[str], str, None] = None,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    preserve: Optional[bool] = False,
    seed: Optional[int] = -1,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> DataNode:
    """
    Executes a Python function.

    This operator can be used to execute custom Python code in the DALI pipeline.
    The function receives the data from DALI as NumPy arrays in case of CPU operators or
    as CuPy arrays for GPU operators. It is expected to return the results in the same format. For
    a more universal data format, see :meth:`nvidia.dali.fn.dl_tensor_python_function`.
    The function should not modify input tensors.
    """
    # This is just a stub of the documentation, for details use help() or visit the html docs.
    ...

@overload
def python_function(
    *input: Union[DataNode, TensorLikeIn],
    function: Callable[..., Union[Any, Tuple[Any, ...], None]],
    batch_processing: bool = False,
    num_outputs: int = 1,
    output_layouts: Union[Sequence[str], str, None] = None,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    preserve: Optional[bool] = False,
    seed: Optional[int] = -1,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[DataNode, Sequence[DataNode], None]:
    """
    Executes a Python function.

    This operator can be used to execute custom Python code in the DALI pipeline.
    The function receives the data from DALI as NumPy arrays in case of CPU operators or
    as CuPy arrays for GPU operators. It is expected to return the results in the same format. For
    a more universal data format, see :meth:`nvidia.dali.fn.dl_tensor_python_function`.
    The function should not modify input tensors.
    """
    # This is just a stub of the documentation, for details use help() or visit the html docs.
    ...

@overload
def dl_tensor_python_function(
    *input: Union[DataNode, TensorLikeIn],
    function: Callable[..., Union[Any, Tuple[Any, ...], None]],
    batch_processing: bool = True,
    synchronize_stream: Optional[bool] = True,
    output_layouts: Union[Sequence[str], str, None] = None,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    preserve: Optional[bool] = False,
    seed: Optional[int] = -1,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> DataNode:
    """
    Executes a Python function that operates on DLPack tensors.

    The function should not modify input tensors.

    For the GPU operator, it is the user's responsibility to synchronize the device code with DALI.
    To synchronize the device code with DALI, synchronize DALI's work before the operator call
    with the `synchronize_stream` flag (enabled by default) and ensure that the scheduled device
    tasks are finished in the operator call. The GPU code can be executed on the CUDA stream used
    by DALI, which can be obtained by calling the ``current_dali_stream()`` function. In this case,
    the `synchronize_stream` flag can be set to False.
    """
    # This is just a stub of the documentation, for details use help() or visit the html docs.
    ...

@overload
def dl_tensor_python_function(
    *input: Union[DataNode, TensorLikeIn],
    function: Callable[..., Union[Any, Tuple[Any, ...], None]],
    batch_processing: bool = True,
    num_outputs: int = 1,
    synchronize_stream: Optional[bool] = True,
    output_layouts: Union[Sequence[str], str, None] = None,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    preserve: Optional[bool] = False,
    seed: Optional[int] = -1,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[DataNode, Sequence[DataNode], None]:
    """
    Executes a Python function that operates on DLPack tensors.

    The function should not modify input tensors.

    For the GPU operator, it is the user's responsibility to synchronize the device code with DALI.
    To synchronize the device code with DALI, synchronize DALI's work before the operator call
    with the `synchronize_stream` flag (enabled by default) and ensure that the scheduled device
    tasks are finished in the operator call. The GPU code can be executed on the CUDA stream used
    by DALI, which can be obtained by calling the ``current_dali_stream()`` function. In this case,
    the `synchronize_stream` flag can be set to False.
    """
    # This is just a stub of the documentation, for details use help() or visit the html docs.
    ...
