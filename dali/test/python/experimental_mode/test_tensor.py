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

import nvidia.dali.experimental.dynamic as ndd
import numpy as np
from nose_utils import SkipTest, attr
from nose2.tools import params
import nvidia.dali.backend as _b
from nose_utils import assert_raises


def asnumpy(tensor):
    return np.array(tensor.cpu().evaluate()._backend)


def test_from_numpy():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    t = ndd.Tensor(data)
    a = np.array(t._backend)
    assert np.array_equal(data, a)


@attr("pytorch")
@params(
    ("cpu", "cpu"),
    ("cpu", "gpu"),
    # ("cuda", "cpu"),   # Disabled due to to mishandling of pinned buffers by torch dlpack
    ("cuda", "gpu"),
)
def test_from_torch(src_device, dst_device):
    import torch

    data = torch.tensor([[1, 2, 3], [4, 5, 6]], device=src_device)
    t = ndd.Tensor(data, device=dst_device).evaluate()
    a = torch.from_dlpack(t._backend)
    if src_device == "cuda":
        a = a.cuda()
    elif src_device == "cpu":
        a = a.cpu()
    assert torch.equal(data, a)


def test_device_change():
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    c1 = ndd.Tensor(data)
    g = c1.gpu().evaluate()
    assert isinstance(g._backend, _b.TensorGPU)
    c2 = g.cpu().evaluate()
    assert isinstance(c2._backend, _b.TensorCPU)
    assert np.array_equal(c2._backend, data)


@attr("multi_gpu")
def test_device_change_multi_gpu():
    if _b.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    c1 = ndd.Tensor(data)
    g1 = c1.gpu(0).evaluate()
    assert isinstance(g1._backend, _b.TensorGPU)
    assert g1._backend.device_id() == 0
    g2 = c1.gpu(1).evaluate()
    assert g2._backend.device_id() == 1
    c2 = g2.cpu().evaluate()
    assert isinstance(c2._backend, _b.TensorCPU)
    assert np.array_equal(c2._backend, data)


@params(("cpu",), ("gpu",))
def test_tensor_converting_constructor(device_type):
    data = np.array([1, 1.25, 2.2, 0x1000001], dtype=np.float64)
    data_i32 = np.int32([1, 1, 2, 0x1000001])
    data_fp32 = np.float32([1, 1.25, 2.2, 0x1000000])
    # loss of precision --------------------------^
    orig = ndd.Tensor(data, device=device_type)
    i32 = ndd.Tensor(data, device=device_type, dtype=ndd.int32)
    fp32 = ndd.Tensor(data, device=device_type, dtype=ndd.float32)
    assert orig.dtype == ndd.float64
    assert orig.device == ndd.Device(device_type)
    assert orig._backend.dtype == ndd.float64.type_id
    assert i32.dtype == ndd.int32
    assert i32.device == ndd.Device(device_type)
    assert i32._backend.dtype == ndd.int32.type_id
    assert fp32.dtype == ndd.float32
    assert fp32.device == ndd.Device(device_type)
    assert fp32._backend.dtype == ndd.float32.type_id
    assert np.array_equal(data, asnumpy(orig))
    assert np.array_equal(data_i32, asnumpy(i32))
    assert np.array_equal(data_fp32, asnumpy(fp32))


@params(("cpu",), ("gpu",))
def test_tensor_makes_copy(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = ndd.Tensor(data, device=device_type)
    dst = ndd.tensor(src._backend)
    assert dst._backend is not src._backend
    assert dst._backend.data_ptr() != src._backend.data_ptr()


@params(("cpu",), ("gpu",))
def test_as_tensor_doesnt_make_copy(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = ndd.Tensor(data, device=device_type)
    dst = ndd.as_tensor(src._backend)
    assert dst._backend.data_ptr() == src._backend.data_ptr()


@params(("cpu",), ("gpu",))
def test_as_tensor_with_conversion(device_type):
    data = np.array([1, 2, 3], dtype=np.int32)
    src = ndd.Tensor(data, device=device_type)
    dst = ndd.as_tensor(src._backend, dtype=ndd.float32)
    assert dst._backend.dtype == ndd.float32.type_id
    assert np.array_equal(data.astype(np.float32), asnumpy(dst))


def test_tensor_clone_properties():
    t = ndd.tensor(np.array([1, 2, 3], dtype=np.int32), layout="X")
    t2 = ndd.tensor(t)
    assert t2.dtype == t.dtype
    assert t2.device == t.device
    assert t2.layout == t.layout
    assert t2.shape == t.shape
    assert t2.size == t.size
    assert t2.nbytes == t.nbytes
    assert t2.itemsize == t.itemsize


def test_scalar():
    scalar = ndd.tensor(5, dtype=ndd.int32)
    assert scalar.ndim == 0
    assert scalar.shape == ()
    assert scalar.dtype == ndd.int32
    assert scalar.device == ndd.Device("cpu")
    assert scalar.layout is None
    assert scalar._backend.dtype == ndd.int32.type_id


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
        tensor = ndd.tensor(np.full(shape, 5, dtype=np.int32))
        assert tensor.ndim == len(shape)
        assert tensor.shape == shape
        assert tensor.size == np.prod(shape)
        assert tensor.nbytes == tensor.size * tensor.dtype.bytes
        assert tensor.itemsize == tensor.dtype.bytes
        assert tensor.dtype == ndd.int32
        assert tensor.device == ndd.Device("cpu")
        assert tensor.layout is None


def test_layout():
    t = ndd.tensor(np.zeros((480, 640, 3), dtype=np.int32))
    assert t.layout is None
    t = ndd.tensor(np.zeros((480, 640, 3), dtype=np.int32), layout="HWC")
    assert t.layout == "HWC"
    with assert_raises(ValueError, glob="dimensions"):
        t = ndd.tensor(np.zeros((480, 640, 3), dtype=np.int32), layout="DHWC")


def test_shape_slice():
    t = ndd.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[1:3, 2:5, 3:7, 4:9]
    assert s.shape == (2, 3, 4, 5)


def test_shape_slices_multi():
    t = ndd.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32), layout="DHWC")
    s = t[1:3, ..., 4:9]
    assert s.layout == "DHWC"
    assert s.shape == (2, 6, 7, 5)
    s = s[:, 2:5, 3:7, :]
    assert s.layout == "DHWC"
    assert s.shape == (2, 3, 4, 5)
    s = s[0, ..., 0]
    assert s.shape == (3, 4)
    assert s.layout == "HW"


def test_shape_slice_dim_removal():
    t = ndd.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32), layout="ABCD")
    s = t[:-1, 0, -2:, 0]
    assert s.shape == (4, 2)
    assert s.layout == "AC"


def test_shape_slice_multi_dim_removal():
    t = ndd.tensor(np.zeros((5, 6, 7, 10), dtype=np.int32))
    s = t[:-1, 0, -2:, 0]
    assert s.shape == (4, 2)
    s = s[:-1, 0]
    assert s.shape == (3,)


def test_tensor_subscript():
    t = ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.int32)
    x = t[1, 1]
    assert asnumpy(x) == 5
    assert x.dtype == ndd.int32


def test_tensor_subscript_negative_step():
    t = ndd.tensor([0, 1, 2, 3, 4, 5], dtype=ndd.int32)
    x = t[-1:1:-1]
    assert x.shape == (4,)
    assert np.array_equal(asnumpy(x), np.int32([5, 4, 3, 2]))

    x = t[::-1]
    assert x.shape == (6,)
    assert np.array_equal(asnumpy(x), np.int32([5, 4, 3, 2, 1, 0]))

    y = x[:-1]
    assert y.shape == (5,)
    assert np.array_equal(asnumpy(y), np.int32([5, 4, 3, 2, 1]))

    z = x[::2]
    assert z.shape == (3,)
    assert np.array_equal(asnumpy(z), np.int32([5, 3, 1]))


def test_tensor_copy_constructor_invocation_result():
    t = ndd.tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), device="cpu")
    t_gpu = t.gpu()  # invocation result - lazy copy
    t_gpu2 = ndd.Tensor(t_gpu)
    assert t_gpu2.device == ndd.Device("gpu")
    assert t_gpu2.dtype == ndd.int32
    assert t_gpu2.shape == (2, 3)
    assert np.array_equal(asnumpy(t_gpu2), asnumpy(t_gpu))
