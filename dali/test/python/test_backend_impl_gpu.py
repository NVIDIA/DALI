# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.backend_impl import *
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
from nose.tools import assert_raises, raises
import cupy as cp
from test_utils import py_buffer_from_address

class ExternalSourcePipe(Pipeline):
  def __init__(self, batch_size, data):
      super(ExternalSourcePipe, self).__init__(batch_size, 1, 0)
      self.output = ops.ExternalSource(device="gpu")
      self.data = data

  def define_graph(self):
      self.out = self.output()
      return self.out

  def iter_setup(self):
      self.feed_input(self.out, self.data)

def test_tensorlist_getitem_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensorlist = pipe.run()[0]
    list_of_tensors = [x for x in tensorlist]

    assert(type(tensorlist[0]) != cp.ndarray)
    assert(type(tensorlist[0]) == TensorGPU)
    assert(type(tensorlist[-3]) == TensorGPU)
    assert(len(list_of_tensors) == len(tensorlist))
    with assert_raises(IndexError):
        tensorlist[len(tensorlist)]
    with assert_raises(IndexError):
        tensorlist[-len(tensorlist) - 1]

def test_data_ptr_tensor_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor = pipe.run()[0][0]
    from_tensor = py_buffer_from_address(tensor.data_ptr(), tensor.shape(), tensor.dtype(), gpu=True)
    # from_tensor is cupy array, convert arr to cupy as well
    assert(cp.allclose(arr[0], from_tensor))

def test_data_ptr_tensor_list_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor_list = pipe.run()[0]
    tensor = tensor_list.as_tensor()
    from_tensor = py_buffer_from_address(tensor_list.data_ptr(), tensor.shape(), tensor.dtype(), gpu=True)
    # from_tensor is cupy array, convert arr to cupy as well
    assert(cp.allclose(arr, from_tensor))

def test_cuda_array_interface_tensor_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor_list = pipe.run()[0]
    assert tensor_list[0].__cuda_array_interface__['data'][0] == tensor_list[0].data_ptr()
    assert tensor_list[0].__cuda_array_interface__['data'][1] == True
    assert np.array_equal(tensor_list[0].__cuda_array_interface__['shape'], tensor_list[0].shape())
    assert tensor_list[0].__cuda_array_interface__['typestr'] == tensor_list[0].dtype()
    assert(cp.allclose(arr[0], cp.asanyarray(tensor_list[0])))

def test_cuda_array_interface_tensor_gpu_create():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor_list = pipe.run()[0]
    assert(cp.allclose(arr[0], cp.asanyarray(tensor_list[0])))

def test_cuda_array_interface_tensor_list_gpu_create():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor_list = pipe.run()[0]
    assert(cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor())))

def test_cuda_array_interface_tensor_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor)))

def test_dlpack_tensor_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr.toDlpack())
    assert(cp.allclose(arr, cp.asanyarray(tensor)))

def test_cuda_array_interface_tensor_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")
    assert(np.allclose(arr.get(), tensor.as_cpu()))

def test_dlpack_tensor_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr.toDlpack(), "NHWC")
    assert(np.allclose(arr.get(), tensor.as_cpu()))

def test_cuda_array_interface_tensor_gpu_to_cpu_device_id():
    arr = cp.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC", 0)
    assert(np.allclose(arr.get(), tensor.as_cpu()))

def test_cuda_array_interface_tensor_list_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor())))

def test_dlpack_tensor_list_gpu_direct_creation():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr.toDlpack(), "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor())))

def test_cuda_array_interface_tensor_list_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC")
    assert(np.allclose(arr.get(), tensor_list.as_cpu().as_tensor()))

def test_dlpack_tensor_list_gpu_to_cpu():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr.toDlpack(), "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor_list.as_tensor())))

def test_cuda_array_interface_tensor_list_gpu_to_cpu_device_id():
    arr = cp.random.rand(3, 5, 6)
    tensor_list = TensorListGPU(arr, "NHWC", 0)
    assert(np.allclose(arr.get(), tensor_list.as_cpu().as_tensor()))

def check_cuda_array_types(t):
    arr = cp.array([[-0.39, 1.5], [-1.5, 0.33]], dtype=t)
    tensor = TensorGPU(arr, "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor)))

def test_cuda_array_interface_types():
    for t in [cp.bool_, cp.int8, cp.int16, cp.int32, cp.int64, cp.uint8,
              cp.uint16, cp.uint32, cp.uint64, cp.float64, cp.float32, cp.float16]:
        yield check_cuda_array_types, t

def check_dlpack_types(t):
    arr = cp.array([[-0.39, 1.5], [-1.5, 0.33]], dtype=t)
    tensor = TensorGPU(arr.toDlpack(), "NHWC")
    assert(cp.allclose(arr, cp.asanyarray(tensor)))

def test_dlpack_interface_types():
    for t in [cp.int8, cp.int16, cp.int32, cp.int64, cp.uint8,
              cp.uint16, cp.uint32, cp.uint64, cp.float64, cp.float32, cp.float16]:
        yield check_dlpack_types, t

@raises(RuntimeError)
def test_cuda_array_interface_tensor_gpu_create_from_numpy():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")

@raises(RuntimeError)
def test_cuda_array_interface_tensor_list_gpu_create_from_numpy():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorGPU(arr, "NHWC")
