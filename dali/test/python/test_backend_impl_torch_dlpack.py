# Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.tensors import TensorCPU, TensorGPU, TensorListCPU, TensorListGPU
import nvidia.dali.tensors as tensors
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import ctypes
from nvidia.dali.backend import CheckDLPackCapsule


def convert_to_torch(tensor, device="cuda", dtype=None, size=None):
    if size is None:
        if isinstance(tensor, TensorListCPU) or isinstance(tensor, TensorListGPU):
            t = tensor.as_tensor()
        else:
            t = tensor
        size = t.shape()
    dali_torch_tensor = torch.empty(size=size, device=device, dtype=dtype)
    c_type_pointer = ctypes.c_void_p(dali_torch_tensor.data_ptr())
    tensor.copy_to_external(c_type_pointer)
    return dali_torch_tensor


def test_dlpack_tensor_gpu_direct_creation():
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        arr = torch.rand(size=[3, 5, 6], device="cuda")
        tensor = TensorGPU(to_dlpack(arr), stream=s.cuda_stream)
        assert int(tensor.stream) == int(s.cuda_stream)
        dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
        assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_tensor_gpu_to_cpu():
    arr = torch.rand(size=[3, 5, 6], device="cuda")
    tensor = TensorGPU(to_dlpack(arr))
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.cpu().eq(dali_torch_tensor.cpu()))


def test_dlpack_tensor_list_gpu_direct_creation():
    arr = torch.rand(size=[3, 5, 6], device="cuda")
    tensor_list = TensorListGPU(to_dlpack(arr), "NHWC")
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_tensor_list_gpu_to_cpu():
    arr = torch.rand(size=[3, 5, 6], device="cuda")
    tensor_list = TensorListGPU(to_dlpack(arr), "NHWC")
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.cpu().eq(dali_torch_tensor.cpu()))


def check_dlpack_types_gpu(t):
    arr = torch.tensor([[0.39, 1.5], [1.5, 0.33]], device="cuda", dtype=t)
    tensor = TensorGPU(to_dlpack(arr), "NHWC")
    dali_torch_tensor = convert_to_torch(
        tensor, device=arr.device, dtype=arr.dtype, size=tensor.shape()
    )
    assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_interface_types():
    for t in [
        # the more recent PyTorch doesn't support
        # torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float64,
        torch.float32,
        torch.float16,
    ]:
        yield check_dlpack_types_gpu, t


def test_dlpack_tensor_cpu_direct_creation():
    arr = torch.rand(size=[3, 5, 6], device="cpu")
    tensor = TensorCPU(to_dlpack(arr))
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_tensor_list_cpu_direct_creation():
    arr = torch.rand(size=[3, 5, 6], device="cpu")
    tensor_list = TensorListCPU(to_dlpack(arr), "NHWC")
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_tensor_list_cpu_direct_creation_list():
    arr = torch.rand(size=[3, 5, 6], device="cpu")
    tensor_list = TensorListCPU([to_dlpack(arr)], "NHWC")
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))


# Check if dlpack tensors behave correctly when created from temporary objects


def test_tensor_cpu_from_dlpack():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a))
        return tensors.TensorCPU(dlt, "")

    out = [create_tmp(i) for i in range(4)]
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t), np.full((4, 4), i))


def test_tensor_list_cpu_from_dlpack():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a))
        return tensors.TensorListCPU(dlt, "")

    out = [create_tmp(i) for i in range(4)]
    for i, tl in enumerate(out):
        np.testing.assert_array_equal(tl.as_array(), np.full((4, 4), i))


def test_tensor_gpu_from_dlpack():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a).cuda())
        return tensors.TensorGPU(dlt, "")

    out = [create_tmp(i) for i in range(4)]
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t.as_cpu()), np.full((4, 4), i))


def test_tensor_list_gpu_from_dlpack():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a).cuda())
        return tensors.TensorListGPU(dlt, "")

    out = [create_tmp(i) for i in range(4)]
    for i, tl in enumerate(out):
        np.testing.assert_array_equal(tl.as_cpu().as_array(), np.full((4, 4), i))


def check_dlpack_types_cpu(t):
    arr = torch.tensor([[0.39, 1.5], [1.5, 0.33]], device="cpu", dtype=t)
    tensor = TensorCPU(to_dlpack(arr), "NHWC")
    dali_torch_tensor = convert_to_torch(
        tensor, device=arr.device, dtype=arr.dtype, size=tensor.shape()
    )
    assert torch.all(arr.eq(dali_torch_tensor))


def test_dlpack_interface_types_cpu():
    for t in [
        # the more recent PyTorch doesn't support
        # torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float64,
        torch.float32,
    ]:
        yield check_dlpack_types_cpu, t


def test_CheckDLPackCapsuleNone():
    info = CheckDLPackCapsule(None)
    assert info == (False, False)


def test_CheckDLPackCapsuleCpu():
    arr = torch.rand(size=[3, 5, 6], device="cpu")
    info = CheckDLPackCapsule(to_dlpack(arr))
    assert info == (True, False)


def test_CheckDLPackCapsuleGpu():
    arr = torch.rand(size=[3, 5, 6], device="cuda")
    info = CheckDLPackCapsule(to_dlpack(arr))
    assert info == (True, True)
