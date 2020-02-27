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
from nose.tools import assert_raises
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
    assert(cp.allclose(cp.array(arr[0]), from_tensor))

def test_data_ptr_tensor_list_gpu():
    arr = np.random.rand(3, 5, 6)
    pipe = ExternalSourcePipe(arr.shape[0], arr)
    pipe.build()
    tensor_list = pipe.run()[0]
    tensor = tensor_list.as_tensor()
    from_tensor = py_buffer_from_address(tensor_list.data_ptr(), tensor.shape(), tensor.dtype(), gpu=True)
    # from_tensor is cupy array, convert arr to cupy as well
    assert(cp.allclose(cp.array(arr), from_tensor))