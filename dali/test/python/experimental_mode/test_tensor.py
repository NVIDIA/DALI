# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as D
import numpy as np
import gc
from nose_utils import SkipTest, attr
from nose2.tools import params
import nvidia.dali.backend as _b


def asnumpy(tensor):
    import numpy as np

    return np.array(tensor.cpu().evaluate()._backend)


def test_from_numpy():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    t = D.Tensor(data)
    a = np.array(t._backend)
    assert np.array_equal(data, a)


@attr("pytorch")
@params(("cpu",), ("cuda",))
def test_from_torch(device_type):
    import torch

    data = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device_type)
    t = D.Tensor(data)
    a = torch.from_dlpack(t._backend)
    assert torch.equal(data, a)


def test_device_change():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    c1 = D.Tensor(data)
    g = c1.gpu().evaluate()
    assert isinstance(g._backend, _b.TensorGPU)
    c2 = g.cpu()
    assert isinstance(c2._backend, _b.TensorCPU)
    assert np.array_equal(c2._backend, data)


@attr("multi_gpu")
def test_device_change_multi_gpu():
    if _b.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    c1 = D.Tensor(data)
    g1 = c1.gpu(0).evaluate()
    assert isinstance(g1._backend, _b.TensorGPU)
    assert g1._backend.device_id() == 0
    g2 = c1.gpu(1).evaluate()
    assert g2._backend.device_id() == 1
    c2 = g2.cpu()
    assert isinstance(c2._backend, _b.TensorCPU)
    assert np.array_equal(c2._backend, data)


@params(("cpu",), ("gpu",))
def test_tensor_cast(device_type):
    data = np.array([1, 1.25, 2.2, 0x1000001], dtype=np.float64)
    orig = D.Tensor(data, device=device_type)
    i32 = D.Tensor(data, device=device_type, dtype=D.int32)
    fp32 = D.Tensor(data, device=device_type, dtype=D.float32)
    assert np.array_equal(data, asnumpy(orig))
    assert np.array_equal(np.int32([1, 1, 2, 0x1000001]), asnumpy(i32))
    assert np.array_equal(np.float32([1, 1.25, 2.2, 0x1000000]), asnumpy(fp32))
