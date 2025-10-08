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
import nvidia.dali.backend as _b
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
            D.tensor(t0),
            D.tensor(t1),
        ],
        device=D.Device(device_type),
    )

    assert isinstance(b, D.Batch)
    assert np.array_equal(asnumpy(b.tensors[0]), t0)
    assert np.array_equal(asnumpy(b.tensors[1]), t1)


def batch_equal(a, b):
    for x, y in zip(a, b, strict=True):
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
    fp32 = D.Batch(orig._backend, device=device_type, dtype=D.float32).evaluate()
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
def test_batch_properties_from_tensor(device_type):
    t = D.tensor(np.array([1, 2, 3], dtype=np.uint8), device=device_type, layout="X")
    b = D.Batch([t])
    assert b.device == D.Device(device_type)
    assert b.dtype == D.uint8
    assert b.layout == "X"
    assert b.batch_size == 1
    assert b.ndim == 1
    assert b.shape == [(3,)]


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


def test_batch_subscript_per_sample():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9], [10, 11, 12]], dtype=D.int32),
        ]
    )
    # unzipped indices (1, 1), (0, 2)
    i = D.as_batch([1, 0])
    j = D.as_batch([1, 2])
    b11 = b.slice[i, j]
    assert isinstance(b11, D.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 9
