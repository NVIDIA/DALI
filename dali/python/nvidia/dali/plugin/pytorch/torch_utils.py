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
from nvidia.dali.tensors import TensorListGPU, TensorListCPU, TensorGPU

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


def to_torch_tensor(tensor_or_tl, device_id=0):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `tensor_or_tl` : TensorGPU or TensorListGPU
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    if isinstance(dali_tensor, (TensorGPU)):
        torch_device = torch.device("cuda", device_id)
    else:
        torch_device = torch.device("cpu")

    out_torch = torch.empty(
        dali_tensor.shape(),
        dtype=to_torch_type[dali_tensor.dtype],
        device=torch_device,
    )

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(out_torch.data_ptr())
    if isinstance(dali_tensor, (TensorGPU)):
        non_blocking = True
        cuda_stream = torch.cuda.current_stream(device=torch_device)
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor_or_tl.copy_to_external(c_type_pointer, stream, non_blocking)
    else:
        tensor_or_tl.copy_to_external(c_type_pointer)

    return out_torch
