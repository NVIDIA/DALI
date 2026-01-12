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
import nvidia.dali.backend as _b
import numpy as np
from nose_utils import attr, SkipTest
from nose2.tools import params
from nose_utils import assert_raises
import test_tensor
import nvidia.dali as dali


def asnumpy(batch_or_tensor):
    if isinstance(batch_or_tensor, ndd.Batch):
        return [test_tensor.asnumpy(t) for t in batch_or_tensor.tensors]
    else:
        return test_tensor.asnumpy(batch_or_tensor)


@params(("cpu",), ("gpu",))
def test_batch_construction(device_type):
    t0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t1 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)

    b = ndd.batch(
        [
            t0,
            t1,
        ],
        device=ndd.Device(device_type),
        layout="AB",
    )

    assert isinstance(b, ndd.Batch)
    assert b.device.device_type == device_type
    assert b.layout == "AB"
    assert np.array_equal(asnumpy(b.tensors[0]), t0)
    assert np.array_equal(asnumpy(b.tensors[1]), t1)
    # check that modifying the original arrays doesn't affect the batch
    t0[0, 0] += 1
    t1[0, 0] += 1
    assert not np.array_equal(asnumpy(b.tensors[0]), t0)
    assert not np.array_equal(asnumpy(b.tensors[1]), t1)

    b.evaluate()
    assert b._storage.layout() == "AB"


@params(("cpu",), ("gpu",))
def test_batch_from_empty_list(device_type):
    with assert_raises(ValueError, glob="Element type"):
        ndd.batch([], device=device_type)
    b = ndd.batch([], dtype=ndd.int32, device=device_type)
    assert isinstance(b, ndd.Batch)
    assert b.dtype == ndd.int32
    assert b.shape == []
    assert b.ndim == 0


@params(("cpu",), ("gpu",))
def test_batch_as_batch(device_type):
    t0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t1 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)

    b0 = ndd.batch(
        [
            t0,
            t1,
        ],
        device=ndd.Device(device_type),
        layout="AB",
    )

    b1 = ndd.batch(b0)
    b2 = ndd.as_batch(b0)
    assert b1.dtype == b0.dtype
    assert b1.shape == b0.shape
    assert b1.layout == b0.layout
    assert b2.dtype == b0.dtype
    assert b2.shape == b0.shape
    assert b2.layout == b0.layout

    assert not b1.tensors[0]._is_same_tensor(b0.tensors[0])
    assert b2.tensors[0]._is_same_tensor(b0.tensors[0])

    b3 = ndd.as_batch(b0, dtype=ndd.float32)
    assert b3.dtype == ndd.float32
    assert b3.shape == b0.shape
    assert b3.layout == b0.layout


def test_broadcast():
    a = np.array([1, 2, 3])
    b = ndd.Batch.broadcast(a, 5).evaluate()
    for i, t in enumerate(b._storage):
        assert np.array_equal(np.array(t), a)


def batch_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not np.array_equal(x, y):
            return False
    return True


@attr("pytorch")
@params(("cpu",), ("cuda",))
def test_batch_construction_with_torch_tensor(device_type):
    import torch

    data = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device_type, dtype=torch.int32)
    b = ndd.as_batch(data)
    assert b.device == ndd.Device("gpu" if device_type == "cuda" else device_type)
    assert b.dtype == ndd.int32
    assert b.batch_size == 2
    assert b.ndim == 1
    assert b.shape == [(3,), (3,)]
    assert b.layout is None
    ref = torch.from_dlpack(ndd.as_tensor(b)._storage)
    assert torch.equal(data[0], ref[0])
    assert torch.equal(data[1], ref[1])


@params(("cpu",), ("gpu",))
def test_batch_construction_with_tensor(device_type):
    t = ndd.tensor(np.array([1, 2, 3], dtype=np.uint8), device=device_type, layout="X")
    b = ndd.Batch([t])
    assert b.device == ndd.Device(device_type)
    assert b.dtype == ndd.uint8
    assert b.layout == "X"
    assert b.batch_size == 1
    assert b.ndim == 1
    assert b.shape == [(3,)]


@params(("cpu",), ("gpu",))
def test_batch_construction_with_conversion(device_type):
    data = [np.float64([1]), np.float64([1.25, 2.2, 0x1000001])]
    data_i32 = [np.int32([1]), np.int32([1, 2, 0x1000001])]
    data_fp32 = [np.float32([1]), np.float32([1.25, 2.2, 0x1000000])]
    # loss of precision --------------------------^
    orig = ndd.Batch(data, device=device_type).evaluate()
    # convert from a list of tensors
    i32 = ndd.Batch(data, device=device_type, dtype=ndd.int32).evaluate()
    # convert from a TensorList object
    fp32 = ndd.Batch(orig._storage, device=device_type, dtype=ndd.float32)
    fp32.evaluate()
    assert orig.dtype == ndd.float64
    assert orig.device == ndd.Device(device_type)
    assert orig._storage.dtype == ndd.float64.type_id
    assert i32.dtype == ndd.int32
    assert i32.device == ndd.Device(device_type)
    assert i32._storage.dtype == ndd.int32.type_id
    assert fp32.dtype == ndd.float32
    assert fp32.device == ndd.Device(device_type)
    assert fp32._storage.dtype == ndd.float32.type_id
    assert batch_equal(data, asnumpy(orig))
    assert batch_equal(data_i32, asnumpy(i32))
    assert batch_equal(data_fp32, asnumpy(fp32))


@params(("cpu",), ("gpu",))
def test_batch_properties_clone(device_type):
    t = ndd.tensor(np.array([1, 2, 3], dtype=np.uint8))
    src = ndd.Batch([t], device=device_type, layout="X")
    b = ndd.batch(src)
    assert b.device == ndd.Device(device_type)
    assert b.dtype == ndd.uint8
    assert b.layout == "X"
    assert b.batch_size == 1
    assert b.ndim == 1
    assert b.shape == [(3,)]


def test_batch_subscript_broadcast():
    b = ndd.as_batch(
        [
            ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.int32),
            ndd.tensor([[7, 8, 9], [10, 11, 12]], dtype=ndd.int32),
        ],
        layout="XY",
    )
    b11 = b.slice[1, 1]
    assert b11.layout is None
    assert b11.dtype == ndd.int32
    assert isinstance(b11, ndd.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 11


def test_batch_partial_slice():
    b = ndd.as_batch(
        [
            ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.int32),
            ndd.tensor([[7, 8, 9], [10, 11, 12]], dtype=ndd.int32),
        ],
        layout="XY",
    )
    b11 = b.slice[..., 1]
    assert b11.layout == "X"
    assert b11.dtype == ndd.int32
    assert isinstance(b11, ndd.Batch)
    assert np.array_equal(asnumpy(b11.tensors[0]), np.array([2, 5], dtype=np.int32))
    assert np.array_equal(asnumpy(b11.tensors[1]), np.array([8, 11], dtype=np.int32))


def test_batch_slice():
    b = ndd.as_batch(
        [
            ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.uint16),
            ndd.tensor([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=ndd.uint16),
        ],
        layout="XY",
    )
    sliced = b.slice[..., 1:-1]
    assert sliced.layout == "XY"
    assert sliced.dtype == ndd.uint16
    assert np.array_equal(asnumpy(sliced.tensors[0]), np.array([[2], [5]], dtype=np.uint16))
    assert np.array_equal(asnumpy(sliced.tensors[1]), np.array([[8, 9], [12, 13]], dtype=np.uint16))


def test_batch_subscript_per_sample():
    b = ndd.as_batch(
        [
            ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.int32),
            ndd.tensor([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=ndd.int32),
        ]
    )
    # unzipped indices (1, 1), (0, 2)
    i = ndd.as_batch([1, 0])
    j = ndd.as_batch([1, 2])
    b11 = b.slice[i, j]
    assert isinstance(b11, ndd.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 9


def test_batch_to_gpu():
    input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t_cpu = ndd.tensor(input)
    t_gpu = t_cpu.gpu()
    assert t_gpu.device == ndd.Device("gpu")
    b_gpu = ndd.Batch([t_gpu])
    b_gpu.evaluate()
    assert b_gpu.device == ndd.Device("gpu")
    assert b_gpu.dtype == ndd.int32
    assert b_gpu.batch_size == 1
    assert b_gpu.ndim == 2
    assert b_gpu.shape == [(2, 3)]
    assert np.array_equal(asnumpy(b_gpu.tensors[0]), input)


@attr("multi_gpu")
def test_cross_device_copy():
    if _b.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    c0 = ndd.as_batch(
        [
            ndd.tensor([[1, 2, 3], [4, 5, 6]], dtype=ndd.int32),
            ndd.tensor([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=ndd.int32),
        ]
    )
    g0 = c0.to_device("gpu:0")
    g1 = g0.to_device("gpu:1")
    c1 = g1.cpu()
    assert batch_equal(asnumpy(c0), asnumpy(c1))
    g0 = g1.to_device("gpu:0")
    c0 = g0.cpu()
    assert batch_equal(asnumpy(c0), asnumpy(c1))


@params(("cpu",), ("gpu",))
def test_batch_from_enum_auto(device_type):
    for value, type in [
        (dali.types.INTERP_CUBIC, ndd.InterpType),
        (dali.types.YCbCr, ndd.ImageType),
        (dali.types.INT32, ndd.DataType),
    ]:
        t = ndd.Batch([value, value], device=device_type)
        assert t.dtype == type
        as_int = ndd.batch(t, dtype=ndd.int32, device="cpu")
        assert as_int.tensors[0].item() == int(value)
        assert as_int.tensors[1].item() == int(value)


@params(("cpu",), ("gpu",))
def test_batch_from_enum_with_dtype(device_type):
    for value, type in [
        (dali.types.INTERP_CUBIC, ndd.InterpType),
        (dali.types.YCbCr, ndd.ImageType),
        (dali.types.INT32, ndd.DataType),
    ]:
        t = ndd.Batch([value, value], device=device_type, dtype=type)
        assert t.dtype == type
        as_int = ndd.batch(t, dtype=ndd.int32, device="cpu")
        assert as_int.tensors[0].item() == int(value)
        assert as_int.tensors[1].item() == int(value)


@params(("cpu",), ("gpu",))
def test_batch_from_enum_value_and_dtype(device_type):
    for value, type in [
        (dali.types.INTERP_CUBIC, ndd.InterpType),
        (dali.types.YCbCr, ndd.ImageType),
        (dali.types.INT32, ndd.DataType),
    ]:
        t = ndd.Batch([int(value), int(value)], device=device_type, dtype=type)
        assert t.dtype == type
        as_int = ndd.batch(t, dtype=ndd.int32, device="cpu")
        assert as_int.tensors[0].item() == int(value)
        assert as_int.tensors[1].item() == int(value)
