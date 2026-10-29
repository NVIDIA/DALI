# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile

import lmdb
import numpy as np
from nvidia.dali import fn, pipeline_def
from nose2.tools import params
from nose_utils import assert_raises


def _bytes_field(number, value):
    # These fixtures contain only short fields, whose lengths fit in one varint byte.
    assert len(value) < 128
    return bytes([number * 8 + 2, len(value)]) + value


def _record(*tensors):
    return b"".join(_bytes_field(1, tensor) for tensor in tensors)


_image = b"\x10\x04" + _bytes_field(6, b"image")
_label = b"\x10\x02" + _bytes_field(4, b"\x00")


def _run_caffe2(record, **kwargs):
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "data.lmdb")
        with lmdb.open(path, map_size=1 << 20) as env:
            with env.begin(write=True) as txn:
                txn.put(b"0", record)

        @pipeline_def(batch_size=1, device_id=None, num_threads=1)
        def pipe():
            return tuple(fn.readers.caffe2(path=[path], **kwargs)[:2])

        return tuple(output.at(0).copy() for output in pipe().run())


@params(
    (b"\x38\x01", {}, "*missing tensor 0*"),
    (_record(_image), {}, "*missing tensor 1*"),
    (_record(b"\x10\x04", _label), {}, "*missing string data*"),
    (_record(b"\x10\x03", _label), {}, "*2 or 3 dimensions*"),
    (_record(b"\x08\x01\x10\x03", _label), {}, "*2 or 3 dimensions*"),
    (_record(b"\x08\x01\x08\x01\x10\x03", _label), {}, "*dimensions exceed*"),
    (_record(_image, _label), {"additional_inputs": 1}, "*missing additional tensors*"),
    (_record(_image, _label), {"label_type": 3}, "*missing tensor 2*"),
    (_record(_image, _label, b"\x10\x01"), {"label_type": 3}, "*one float value*"),
)
def test_caffe2_rejects_incomplete_tensor_fields(record, kwargs, message):
    assert_raises(RuntimeError, _run_caffe2, record, glob=message, **kwargs)


def test_caffe2_valid_encoded_image():
    image, label = _run_caffe2(_record(_image, _label))
    np.testing.assert_array_equal(image, np.frombuffer(b"image", dtype=np.uint8))
    np.testing.assert_array_equal(label, [0])


def test_caffe2_valid_byte_image():
    tensor = b"\x08\x01\x08\x02\x10\x03" + _bytes_field(5, b"ab")
    image, label = _run_caffe2(_record(tensor, _label))
    np.testing.assert_array_equal(image, np.array([[[97], [98]]], dtype=np.uint8))
    np.testing.assert_array_equal(label, [0])
