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
from nose_utils import assert_raises


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
def test_tensor_converting_constructor(device_type):
    data = np.array([1, 1.25, 2.2, 0x1000001], dtype=np.float64)
    data_i32 = np.int32([1, 1, 2, 0x1000001])
    data_fp32 = np.float32([1, 1.25, 2.2, 0x1000000])
    # loss of precision --------------------------^
    orig = D.Tensor(data, device=device_type)
    i32 = D.Tensor(data, device=device_type, dtype=D.int32)
    fp32 = D.Tensor(data, device=device_type, dtype=D.float32)
    assert orig.dtype == D.float64
    assert orig.device == D.Device(device_type)
    assert orig._backend.dtype == D.float64.type_id
    assert i32.dtype == D.int32
    assert i32.device == D.Device(device_type)
    assert i32._backend.dtype == D.int32.type_id
    assert fp32.dtype == D.float32
    assert fp32.device == D.Device(device_type)
    assert fp32._backend.dtype == D.float32.type_id
    assert np.array_equal(data, asnumpy(orig))
    assert np.array_equal(data_i32, asnumpy(i32))
    assert np.array_equal(data_fp32, asnumpy(fp32))


@params(("cpu",), ("gpu",))
def test_tensor_makes_copy(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = D.Tensor(data, device=device_type)
    dst = D.tensor(src._backend)
    assert dst._backend is not src._backend
    assert dst._backend.data_ptr() != src._backend.data_ptr()


@params(("cpu",), ("gpu",))
def test_as_tensor_doesnt_make_copy(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = D.Tensor(data, device=device_type)
    dst = D.as_tensor(src._backend)
    assert dst._backend.data_ptr() == src._backend.data_ptr()


@params(("cpu",), ("gpu",))
def test_as_tensor_doesnt_make_copy(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = D.Tensor(data, device=device_type)
    dst = D.as_tensor(src._backend)
    assert dst._backend.data_ptr() == src._backend.data_ptr()


@params(("cpu",), ("gpu",))
def test_as_tensor_with_conversion(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = D.Tensor(data, device=device_type)
    dst = D.as_tensor(src._backend, dtype=D.float32)
    assert dst._backend.dtype == D.float32.type_id
    assert np.array_equal(data.astype(np.float32), asnumpy(dst))


def test_scalar():
    scalar = D.tensor(5, dtype=D.int32)
    assert scalar.ndim == 0
    assert scalar.shape == ()
    assert scalar.dtype == D.int32
    assert scalar.device == D.Device("cpu")
    assert scalar.layout is None
    assert scalar._backend.dtype == D.int32.type_id


def test_shapes():
    shapes = [
        (),
        (1,),
        (42,),
        (480, 640, 3),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6, 7),
        (1, 2, 3, 4, 5, 6, 7, 6),
        (1, 2, 3, 4, 5, 6, 7, 6, 5),
        (1, 2, 3, 4, 5, 6, 7, 6, 5, 4),
    ]
    for shape in shapes:
        tensor = D.tensor(np.full(shape, 5, dtype=np.int32))
        assert tensor.ndim == len(shape)
        assert tensor.shape == shape
        assert tensor.size == np.prod(shape)
        assert tensor.nbytes == tensor.size * tensor.dtype.bytes
        assert tensor.itemsize == tensor.dtype.bytes
        assert tensor.dtype == D.int32
        assert tensor.device == D.Device("cpu")
        assert tensor.layout is None


def test_layout():
    t = D.tensor(np.zeros((480, 640, 3), dtype=np.int32))
    assert t.layout is None
    t = D.tensor(np.zeros((480, 640, 3), dtype=np.int32), layout="HWC")
    assert t.layout == "HWC"
    with assert_raises(ValueError, glob="dimensions"):
        t = D.tensor(np.zeros((480, 640, 3), dtype=np.int32), layout="DHWC")


def test_shape_slice():
    t = D.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[1:3, 2:5, 3:7, 4:9]
    print(s.shape)
    assert s.shape == (2, 3, 4, 5)


def test_shape_slices_multi():
    t = D.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[1:3, ..., 4:9]
    assert s.shape == (2, 6, 7, 5)
    s = s[:, 2:5, 3:7, :]
    assert s.shape == (2, 3, 4, 5)


def test_shape_slice_dim_removal():
    t = D.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[:-1, 0, -2:, 0]
    assert s.shape == (4, 2)


def test_shape_slice_multi_dim_removal():
    t = D.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[:-1, 0, -2:, 0]
    assert s.shape == (4, 2)
    s = s[:-1, 0]
    assert s.shape == (3,)


def test_tensor_subscript():
    t = D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32)
    assert asnumpy(t[1, 1]) == 5


def test_batch_subscript_broadcast():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9], [10, 11, 12]], dtype=D.int32),
        ]
    )
    b11 = b.slice[1, 1]
    assert isinstance(b11, D.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 11
