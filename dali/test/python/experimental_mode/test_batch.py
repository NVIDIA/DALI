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
from nose_utils import attr
from nose2.tools import params
from nose_utils import assert_raises
import test_tensor


def asnumpy(batch_or_tensor):
    if isinstance(batch_or_tensor, D.Batch):
        return [test_tensor.asnumpy(t) for t in batch_or_tensor.tensors]
    else:
        return test_tensor.asnumpy(batch_or_tensor)


@params(("cpu",), ("gpu",))
def test_batch_construction(device_type):
    t0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t1 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)

    b = D.batch(
        [
            t0,
            t1,
        ],
        device=D.Device(device_type),
        layout="AB",
    )

    assert isinstance(b, D.Batch)
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
    assert b._backend.layout() == "AB"


@params(("cpu",), ("gpu",))
def test_batch_from_empty_list(device_type):
    with assert_raises(ValueError, glob="Element type"):
        D.batch([], device=device_type)
    b = D.batch([], dtype=D.int32, device=device_type)
    assert b.dtype == D.int32
    assert b.shape == []
    assert b.ndim == 0


@params(("cpu",), ("gpu",))
def test_batch_as_batch(device_type):
    t0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t1 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)

    b0 = D.batch(
        [
            t0,
            t1,
        ],
        device=D.Device(device_type),
        layout="AB",
    )

    b1 = D.batch(b0)
    b2 = D.as_batch(b0)
    assert b1.dtype == b0.dtype
    assert b1.shape == b0.shape
    assert b1.layout == b0.layout
    assert b2.dtype == b0.dtype
    assert b2.shape == b0.shape
    assert b2.layout == b0.layout

    assert not b1.tensors[0]._is_same_tensor(b0.tensors[0])
    assert b2.tensors[0]._is_same_tensor(b0.tensors[0])

    b3 = D.as_batch(b0, dtype=D.float32)
    assert b3.dtype == D.float32
    assert b3.shape == b0.shape
    assert b3.layout == b0.layout


def test_broadcast():
    a = np.array([1, 2, 3])
    b = D.Batch.broadcast(a, 5).evaluate()
    for i, t in enumerate(b._backend):
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
    b = D.as_batch(data)
    assert b.device == D.Device("gpu" if device_type == "cuda" else device_type)
    assert b.dtype == D.int32
    assert b.batch_size == 2
    assert b.ndim == 1
    assert b.shape == [(3,), (3,)]
    assert b.layout is None
    ref = torch.from_dlpack(D.as_tensor(b)._backend)
    assert torch.equal(data[0], ref[0])
    assert torch.equal(data[1], ref[1])


@params(("cpu",), ("gpu",))
def test_batch_construction_with_tensor(device_type):
    t = D.tensor(np.array([1, 2, 3], dtype=np.uint8), device=device_type, layout="X")
    b = D.Batch([t])
    assert b.device == D.Device(device_type)
    assert b.dtype == D.uint8
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
    orig = D.Batch(data, device=device_type).evaluate()
    # convert from a list of tensors
    i32 = D.Batch(data, device=device_type, dtype=D.int32).evaluate()
    # convert from a TensorList object
    fp32 = D.Batch(orig._backend, device=device_type, dtype=D.float32)
    fp32.evaluate()
    assert orig.dtype == D.float64
    assert orig.device == D.Device(device_type)
    assert orig._backend.dtype == D.float64.type_id
    assert i32.dtype == D.int32
    assert i32.device == D.Device(device_type)
    assert i32._backend.dtype == D.int32.type_id
    assert fp32.dtype == D.float32
    assert fp32.device == D.Device(device_type)
    assert fp32._backend.dtype == D.float32.type_id
    assert batch_equal(data, asnumpy(orig))
    assert batch_equal(data_i32, asnumpy(i32))
    assert batch_equal(data_fp32, asnumpy(fp32))


@params(("cpu",), ("gpu",))
def test_batch_properties_clone(device_type):
    t = D.tensor(np.array([1, 2, 3], dtype=np.uint8))
    src = D.Batch([t], device=device_type, layout="X")
    b = D.batch(src)
    assert b.device == D.Device(device_type)
    assert b.dtype == D.uint8
    assert b.layout == "X"
    assert b.batch_size == 1
    assert b.ndim == 1
    assert b.shape == [(3,)]


def test_batch_subscript_broadcast():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9], [10, 11, 12]], dtype=D.int32),
        ],
        layout="XY",
    )
    b11 = b.slice[1, 1]
    assert b11.layout is None
    assert b11.dtype == D.int32
    assert isinstance(b11, D.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 11


def test_batch_partial_slice():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9], [10, 11, 12]], dtype=D.int32),
        ],
        layout="XY",
    )
    b11 = b.slice[..., 1]
    assert b11.layout == "X"
    assert b11.dtype == D.int32
    assert isinstance(b11, D.Batch)
    assert np.array_equal(asnumpy(b11.tensors[0]), np.array([2, 5], dtype=np.int32))
    assert np.array_equal(asnumpy(b11.tensors[1]), np.array([8, 11], dtype=np.int32))


def test_batch_slice():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.uint16),
            D.tensor([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=D.uint16),
        ],
        layout="XY",
    )
    sliced = b.slice[..., 1:-1]
    assert sliced.layout == "XY"
    assert sliced.dtype == D.uint16
    assert np.array_equal(asnumpy(sliced.tensors[0]), np.array([[2], [5]], dtype=np.uint16))
    assert np.array_equal(asnumpy(sliced.tensors[1]), np.array([[8, 9], [12, 13]], dtype=np.uint16))


def test_batch_subscript_per_sample():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=D.int32),
        ]
    )
    # unzipped indices (1, 1), (0, 2)
    i = D.as_batch([1, 0])
    j = D.as_batch([1, 2])
    b11 = b.slice[i, j]
    assert isinstance(b11, D.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 9


def test_batch_to_gpu():
    input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t_cpu = D.tensor(input)
    t_gpu = t_cpu.gpu()
    assert t_gpu.device == D.Device("gpu")
    b_gpu = D.Batch([t_gpu])
    b_gpu.evaluate()
    assert b_gpu.device == D.Device("gpu")
    assert b_gpu.dtype == D.int32
    assert b_gpu.batch_size == 1
    assert b_gpu.ndim == 2
    assert b_gpu.shape == [(2, 3)]
    assert np.array_equal(asnumpy(b_gpu.tensors[0]), input)
