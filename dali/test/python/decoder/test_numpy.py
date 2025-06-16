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

import io

import numpy as np
import nvidia.dali.fn as fn
from nose_utils import raises
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType


def box_source(data: list[np.ndarray]):
    for sample in data:
        buffer = io.BytesIO()
        np.save(buffer, sample, allow_pickle=False)
        buffer.seek(0)
        buff = np.frombuffer(buffer.read(), dtype=np.uint8)
        yield (buff,)


def test_normal_decoding():
    num_samples = 8
    data = [np.random.normal(size=(2, s)).astype(np.float32) for s in range(num_samples)]

    @pipeline_def(batch_size=1, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()

    for i in range(num_samples):
        output = p.run()
        test = np.asarray(output[0][0])
        assert np.all(test == data[i])


def test_fortran_decoding():
    num_samples = 8
    data = [np.array(np.random.normal(size=(2, s)), order="F") for s in range(num_samples)]

    @pipeline_def(batch_size=1, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()

    for i in range(num_samples):
        output = p.run()
        test = np.asarray(output[0][0])
        assert np.all(test == data[i])


def test_variable_shape():
    num_samples = 8
    data = [np.random.normal(size=(2, s)).astype(np.float32) for s in range(num_samples)]

    @pipeline_def(batch_size=num_samples, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()

    output = p.run()
    for i in range(num_samples):
        test = np.asarray(output[0][i])
        assert np.all(test == data[i])


def test_casting_decoding():
    num_samples = 8
    data = [
        np.random.normal(loc=0, scale=255, size=(2, s)).astype(np.int32) for s in range(num_samples)
    ]

    @pipeline_def(batch_size=1, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy, dtype=DALIDataType.FLOAT)
        return decoded

    p = pipe()
    p.build()

    for i in range(num_samples):
        output = p.run()
        test = np.asarray(output[0][0])
        assert np.all(test == data[i].astype(np.float32))


@raises(RuntimeError, "All samples in the batch must have the same number of dimensions")
def test_raise_different_dims():
    data = [np.empty((2, 3), dtype=np.float32), np.empty((2, 3, 4), dtype=np.float32)]

    @pipeline_def(batch_size=2, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    p.run()


@raises(RuntimeError, "Got bad magic string for numpy header")
def test_raise_bad_numpy_magic():
    def bad_source():
        yield (np.zeros(100, dtype=np.uint8),)

    @pipeline_def(batch_size=1, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=bad_source(), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    p.run()


@raises(RuntimeError, "All samples in the batch must have the same data type")
def test_raise_different_dtype_without_arg():
    data = [np.empty((2, 3), dtype=np.float32), np.empty((2, 3), dtype=np.int32)]

    @pipeline_def(batch_size=2, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    p.run()


def test_different_dtype_with_cast():
    data = [np.full((2, 3), 5, dtype=np.float32), np.full((2, 3), 5, dtype=np.int32)]

    @pipeline_def(batch_size=2, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy, dtype=DALIDataType.FLOAT)
        return decoded

    p = pipe()
    p.build()
    a, b = p.run()[0]
    assert np.all(np.array(a) == np.array(b))
