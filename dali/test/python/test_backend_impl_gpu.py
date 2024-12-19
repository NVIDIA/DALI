# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.backend_impl import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.tensors as tensors
import nvidia.dali.types as types
import numpy as np
from nose_utils import assert_raises, raises
import cupy as cp
from test_utils import py_buffer_from_address


class ExternalSourcePipe(Pipeline):
    def __init__(self, batch_size, data, use_copy_kernel=False):
        super(ExternalSourcePipe, self).__init__(batch_size, 1, 0)
        self.output = ops.ExternalSource(device="gpu")
        self.data = data
        self.use_copy_kernel = use_copy_kernel

    def define_graph(self):
        self.out = self.output()
        return self.out

    def iter_setup(self):
        self.feed_input(self.out, self.data, use_copy_kernel=self.use_copy_kernel)


def test_tensorlist_getitem_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensorlist = pipe.run()[0]
    list_of_tensors = [x for x in tensorlist]

    assert type(tensorlist[0]) is not cp.ndarray
    assert type(tensorlist[0]) is TensorGPU
    assert type(tensorlist[-3]) is TensorGPU
    assert len(list_of_tensors) == len(tensorlist)
    with assert_raises(IndexError, glob="TensorListCPU index out of range"):
        tensorlist[len(tensorlist)]
    with assert_raises(IndexError, glob="TensorListCPU index out of range"):
        tensorlist[-len(tensorlist) - 1]


def test_data_ptr_tensor_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensor = pipe.run()[0][0]
    from_tensor = py_buffer_from_address(
        tensor.data_ptr(), tensor.shape(), types.to_numpy_type(tensor.dtype), gpu=True
    )
    # from_tensor is cupy array, convert arr to cupy as well
    assert cp.allclose(arr[0], from_tensor)


def test_data_ptr_tensor_list_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensor_list = pipe.run()[0]
    tensor = tensor_list.as_tensor()
    from_tensor = py_buffer_from_address(
        tensor_list.data_ptr(), tensor.shape(), types.to_numpy_type(tensor.dtype), gpu=True
    )
    # from_tensor is cupy array, convert arr to cupy as well
    assert cp.allclose(arr, from_tensor)


def test_cuda_array_interface_tensor_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensor_list = pipe.run()[0]
    assert tensor_list[0].__cuda_array_interface__["data"][0] == tensor_list[0].data_ptr()
    assert not tensor_list[0].__cuda_array_interface__["data"][1]
    assert np.array_equal(tensor_list[0].__cuda_array_interface__["shape"], tensor_list[0].shape())
    type_str = tensor_list[0].__cuda_array_interface__["typestr"]
    dtype = types.to_numpy_type(tensor_list[0].dtype)
    assert np.dtype(type_str) == np.dtype(dtype)
    assert cp.allclose(arr[0], cp.asanyarray(tensor_list[0]))


def test_cuda_array_interface_tensor_gpu_create():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensor_list = pipe.run()[0]
    assert cp.allclose(arr[0], cp.asanyarray(tensor_list[0]))


def test_cuda_array_interface_tensor_list_gpu_create():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    tensor_list = pipe.run()[0]
    assert cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_tensor_gpu_create_copy_kernel():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr, use_copy_kernel=True)
    tensor_list = pipe.run()[0]
    assert cp.allclose(arr[0], cp.asanyarray(tensor_list[0]))


def test_cuda_array_interface_tensor_list_gpu_create_copy_kernel():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr, use_copy_kernel=True)
    tensor_list = pipe.run()[0]
    assert cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_tensor_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor))


def test_dlpack_tensor_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr.toDlpack())
    assert cp.allclose(arr, cp.asanyarray(tensor))


def test_cuda_array_interface_tensor_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")
    assert np.allclose(arr.get(), tensor.as_cpu())


def test_dlpack_tensor_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr.toDlpack(), "NHWC")
    assert np.allclose(arr.get(), tensor.as_cpu())


def test_cuda_array_interface_tensor_gpu_to_cpu_device_id():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC", 0)
    assert np.allclose(arr.get(), tensor.as_cpu())


def test_cuda_array_interface_tensor_list_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_tensor_list_gpu_direct_creation_list():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU([arr], "NHWC")
    assert cp.allclose(arr.reshape(tuple([1]) + arr.shape), cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_v3_stream():
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    from pycuda.tools import clear_context_caches, make_default_context

    cuda.init()
    context = make_default_context()
    test_input = np.random.randn(4, 4).astype(np.float32)
    g = gpuarray.to_gpu(test_input)
    TensorGPU(g)
    context.pop()
    clear_context_caches()


def test_dlpack_tensor_list_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr.toDlpack(), "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor()))


def test_dlpack_tensor_list_gpu_direct_creation_list():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU([arr.toDlpack()], "NHWC")
    assert cp.allclose(arr.reshape(tuple([1]) + arr.shape), cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_tensor_list_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC")
    assert np.allclose(arr.get(), tensor_list.as_cpu().as_tensor())


def test_dlpack_tensor_list_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr.toDlpack(), "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor()))


def test_cuda_array_interface_tensor_list_gpu_to_cpu_device_id():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC", 0)
    assert np.allclose(arr.get(), tensor_list.as_cpu().as_tensor())


def check_cuda_array_types(t):
    arr = cp.array([[-0.39, 1.5], [-1.5, 0.33]], dtype=t)
    tensor = TensorGPU(arr, "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor))


def test_cuda_array_interface_types():
    for t in [
        cp.bool_,
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float64,
        cp.float32,
        cp.float16,
    ]:
        yield check_cuda_array_types, t


def check_dlpack_types(t):
    arr = cp.array([[-0.39, 1.5], [-1.5, 0.33]], dtype=t)
    tensor = TensorGPU(arr.toDlpack(), "NHWC")
    assert cp.allclose(arr, cp.asanyarray(tensor))


def test_dlpack_interface_types():
    for t in [
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float64,
        cp.float32,
        cp.float16,
    ]:
        yield check_dlpack_types, t


@raises(RuntimeError, glob="Provided object doesn't support cuda array interface protocol.")
def test_cuda_array_interface_tensor_gpu_create_from_numpy():
    arr = np.random.rand(3, 5, 6)
    TensorGPU(arr, "NHWC")


@raises(RuntimeError, glob="Provided object doesn't support cuda array interface protocol.")
def test_cuda_array_interface_tensor_list_gpu_create_from_numpy():
    arr = np.random.rand(3, 5, 6)
    TensorGPU(arr, "NHWC")


def test_tensor_gpu_squeeze():
    def check_squeeze(shape, dim, in_layout, expected_out_layout):
        arr = cp.random.rand(*shape)
        t = TensorGPU(arr, in_layout)
        is_squeezed = t.squeeze(dim)
        should_squeeze = len(expected_out_layout) < len(in_layout)
        arr_squeeze = arr.squeeze(dim)
        t_shape = tuple(t.shape())
        assert t_shape == arr_squeeze.shape, f"{t_shape} != {arr_squeeze.shape}"
        assert t.layout() == expected_out_layout, f"{t.layout()} != {expected_out_layout}"
        assert cp.allclose(arr_squeeze, cp.asanyarray(t))
        assert is_squeezed == should_squeeze, f"{is_squeezed} != {should_squeeze}"

    for dim, shape, in_layout, expected_out_layout in [
        (None, (3, 5, 6), "ABC", "ABC"),
        (None, (3, 1, 6), "ABC", "AC"),
        (1, (3, 1, 6), "ABC", "AC"),
        (-2, (3, 1, 6), "ABC", "AC"),
        (None, (1, 1, 6), "ABC", "C"),
        (1, (1, 1, 6), "ABC", "AC"),
        (None, (1, 1, 1), "ABC", ""),
        (None, (1, 5, 1), "ABC", "B"),
        (-1, (1, 5, 1), "ABC", "AB"),
        (0, (1, 5, 1), "ABC", "BC"),
        (None, (3, 5, 1), "ABC", "AB"),
    ]:
        yield check_squeeze, shape, dim, in_layout, expected_out_layout


# Those tests verify that the Tensor[List]Cpu/Gpu created in Python in a similar fashion
# to how ExternalSource for samples operates keep the data alive.

# The Tensor[List] take the pointer to data and store the reference to buffer/object that owns
# the data to keep the refcount positive while the Tensor[List] lives.
# Without this behavior there was observable bug with creating several temporary
# buffers in the loop and DALI not tracking references to them


def test_tensor_cpu_from_numpy():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        return tensors.TensorCPU(a, "")

    out = [create_tmp(i) for i in range(4)]
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t), np.full((4, 4), i))


def test_tensor_list_cpu_from_numpy():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        return tensors.TensorListCPU(a, "")

    out = [create_tmp(i) for i in range(4)]
    for i, tl in enumerate(out):
        np.testing.assert_array_equal(tl.as_array(), np.full((4, 4), i))


def test_tensor_from_tensor_list_cpu():
    def create_tl(idx):
        a = np.full((3, 4), idx)
        return tensors.TensorListCPU(a, "")

    out = []
    for i in range(5):
        ts = [t for t in create_tl(i)]
        out += ts
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t), np.full((4,), i // 3))


def test_tensor_gpu_from_cupy():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        a_gpu = cp.array(a, dtype=a.dtype)
        return tensors.TensorGPU(a_gpu, "")

    out = [create_tmp(i) for i in range(4)]
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t.as_cpu()), np.full((4, 4), i))


def test_tensor_list_gpu_from_cupy():
    def create_tmp(idx):
        a = np.full((4, 4), idx)
        a_gpu = cp.array(a, dtype=a.dtype)
        return tensors.TensorListGPU(a_gpu, "")

    out = [create_tmp(i) for i in range(4)]
    for i, tl in enumerate(out):
        for j in range(4):
            np.testing.assert_array_equal(np.array(tl[j].as_cpu()), np.full(tl[j].shape(), i))
        np.testing.assert_array_equal(tl.as_cpu().as_array(), np.full((4, 4), i))


def test_tensor_from_tensor_list_gpu():
    def create_tl(idx):
        a = np.full((3, 4), idx)
        a_gpu = cp.array(a, dtype=a.dtype)
        return tensors.TensorListGPU(a_gpu, "")

    out = []
    for i in range(5):
        ts = [t for t in create_tl(i)]
        out += ts
    for i, t in enumerate(out):
        np.testing.assert_array_equal(np.array(t.as_cpu()), np.full((4,), i // 3))


def test_tensor_dlpack_export():
    arr = cp.arange(20)
    tensor = TensorGPU(arr, "NHWC")

    arr_from_dlpack = cp.from_dlpack(tensor)

    assert cp.array_equal(arr, arr_from_dlpack)
