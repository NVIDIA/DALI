# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import nvidia.dali as dali
import numpy as np
import cupy as cp

# This test verifies that the Tensor[List]Cpu/Gpu created in Python in a similar fashion
# to how ExternalSource for samples operates keep the data alive.

# The Tensor[List] take the pointer to data and store the reference to buffer/object that owns
# the data to keep the refcount positive while the Tensor[List] lives.
# Without this behaviour there was observable bug with creating several temporary
# buffers in the loop and DALI not tracking references to them

def test_tensor_cpu_from_numpy():
    def create_tmp(idx):
        a = np.zeros((4, 4)) + idx
        return dali.tensors.TensorCPU(a, "")
    out = [create_tmp(i) for i in range(4)]
    for i, d in enumerate(out):
        np.testing.assert_array_equal(np.array(d), np.zeros((4, 4)) + i)

def test_tensor_list_cpu_from_numpy():
    def create_tmp(idx):
        a = np.zeros((4, 4)) + idx
        return dali.tensors.TensorListCPU(a, "")
    out = [create_tmp(i) for i in range(4)]
    for i, d in enumerate(out):
        np.testing.assert_array_equal(d.as_array(), np.zeros((4, 4)) + i)

def test_tensor_gpu_from_cupy():
    def create_tmp(idx):
        a = np.zeros((4, 4)) + idx
        a_gpu = cp.array(a, dtype=a.dtype)
        return dali.tensors.TensorGPU(a_gpu, "")
    out = [create_tmp(i) for i in range(4)]
    for i, d in enumerate(out):
        np.testing.assert_array_equal(np.array(d.as_cpu()), np.zeros((4, 4)) + i)

def test_tensor_list_gpu_from_cupy():
    def create_tmp(idx):
        a = np.zeros((4, 4)) + idx
        a_gpu = cp.array(a, dtype=a.dtype)
        return dali.tensors.TensorListGPU(a_gpu, "")
    out = [create_tmp(i) for i in range(4)]
    for i, d in enumerate(out):
        np.testing.assert_array_equal(d.as_cpu().as_array(), np.zeros((4, 4)) + i)
