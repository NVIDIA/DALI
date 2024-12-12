# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import ctypes
from nvidia.dali import types
from nvidia.dali.tensors import TensorListGPU, TensorListCPU, TensorGPU, TensorCPU
from typing import Union, Any

to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.BOOL: torch.bool,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64,
}


def feed_ndarray(
    dali_tensor: Union[TensorCPU, TensorGPU, TensorListCPU, TensorListGPU],
    arr: torch.Tensor,
    cuda_stream: Union[torch.cuda.Stream, Any, None] = None,
) -> torch.Tensor:
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    dali_tensor : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    arr : torch.Tensor
            Destination of the copy
    cuda_stream : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, (
        "The element type of DALI Tensor/TensorList"
        " doesn't match the element type of the target PyTorch Tensor: "
        "{} vs {}".format(dali_type, arr.dtype)
    )
    assert dali_tensor.shape() == list(
        arr.size()
    ), "Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".format(
        dali_tensor.shape(), list(arr.size())
    )

    non_blocking = cuda_stream is not None
    cuda_stream = types._raw_cuda_stream_ptr(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, cuda_stream, non_blocking=non_blocking)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


def to_torch_tensor(dali_tensor, copy):
    """
    Converts to torch.Tensor, either copying the data or using dlpack
    """
    if copy:
        torch_dtype = to_torch_type[dali_tensor.dtype]
        if isinstance(dali_tensor, TensorGPU):
            torch_device = torch.device("cuda", dali_tensor.device_id())
        else:
            torch_device = torch.device("cpu")
        torch_output = torch.empty(
            dali_tensor.shape(),
            dtype=torch_dtype,
            device=torch_device,
        )
        cuda_stream = (
            torch.cuda.current_stream(device=torch_device)
            if isinstance(dali_tensor, TensorGPU)
            else None
        )
        feed_ndarray(dali_tensor, torch_output, cuda_stream=cuda_stream)
        return torch_output
    else:
        return torch.from_dlpack(dali_tensor)
