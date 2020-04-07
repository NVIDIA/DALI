# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import assert_raises
from test_utils import py_buffer_from_address


def test_create_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    assert_array_equal(arr, np.array(tensor))

def test_create_tensorlist():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert_array_equal(arr, tensorlist.as_array())

def test_create_tensorlist_as_tensor():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    tensor = tensorlist.as_tensor()
    assert_array_equal(np.array(tensor), tensorlist.as_array())

def test_empty_tensor_tensorlist():
    arr = np.array([], dtype=np.float32)
    tensor = TensorCPU(arr, "NHWC")
    tensorlist = TensorListCPU(arr, "NHWC")
    assert_array_equal(np.array(tensor), tensorlist.as_array())
    assert(np.array(tensor).shape == (0,))
    assert(tensorlist.as_array().shape == (0,))

def test_tensorlist_getitem_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    list_of_tensors = [x for x in tensorlist]

    assert(type(tensorlist.at(0)) == np.ndarray)
    assert(type(tensorlist[0]) != np.ndarray)
    assert(type(tensorlist[0]) == TensorCPU)
    assert(type(tensorlist[-3]) == TensorCPU)
    assert(len(list_of_tensors) == len(tensorlist))
    with assert_raises(IndexError):
        tensorlist[len(tensorlist)]
    with assert_raises(IndexError):
        tensorlist[-len(tensorlist) - 1]

def test_data_ptr_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    from_tensor = py_buffer_from_address(tensor.data_ptr(), tensor.shape(), tensor.dtype())
    assert(np.array_equal(arr, from_tensor))


def test_data_ptr_tensor_list_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    tensor = tensorlist.as_tensor()
    from_tensor_list = py_buffer_from_address(tensorlist.data_ptr(), tensor.shape(), tensor.dtype())
    assert(np.array_equal(arr, from_tensor_list))

def test_array_interface_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert tensorlist[0].__array_interface__['data'][0] == tensorlist[0].data_ptr()
    assert tensorlist[0].__array_interface__['data'][1] == True
    assert np.array_equal(tensorlist[0].__array_interface__['shape'], tensorlist[0].shape())
    assert tensorlist[0].__array_interface__['typestr'] == tensorlist[0].dtype()

#if 0  // TODO(spanev): figure out which return_value_policy to choose
#def test_tensorlist_getitem_slice():
#    arr = np.random.rand(3, 5, 6)
#    tensorlist = TensorListCPU(arr, "NHWC")
#    two_first_tensors = tensorlist[0:2]
#    assert(type(two_first_tensors) == tuple)
#    assert(type(two_first_tensors[0]) == TensorCPU)
