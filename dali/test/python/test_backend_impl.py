# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

def test_create_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, types.NHWC)
    assert_array_equal(arr, np.array(tensor))

def test_create_tensor_list():
    arr = np.random.rand(3, 5, 6)
    tensor_list = TensorListCPU(arr, types.NHWC)
    assert_array_equal(arr, tensor_list.as_array())

def test_create_tensor_list_as_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor_list = TensorListCPU(arr, types.NHWC)
    tensor = tensor_list.as_tensor()
    assert_array_equal(np.array(tensor), tensor_list.as_array())

def test_empty_tensor_tensor_list():
    arr = np.array([], dtype=np.float32)
    tensor = TensorCPU(arr, types.NHWC)
    tensor_list = TensorListCPU(arr, types.NHWC)
    assert_array_equal(np.array(tensor), tensor_list.as_array())
    assert(np.array(tensor).shape == (0,))
    assert(tensor_list.as_array().shape == (0,))