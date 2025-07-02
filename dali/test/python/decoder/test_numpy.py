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
from nose2.tools import params
from nose_utils import raises
from nvidia.dali import fn, pipeline_def
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
        assert np.array_equal(test, data[i])


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
        assert np.array_equal(test, data[i])


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
        assert np.array_equal(test, data[i].astype(np.float32))


@raises(RuntimeError, "All samples in the dataset must have the same number of dimensions")
@params(1, 2)
def test_raise_different_dims(batch_size):
    data = [np.empty(shape, dtype=np.float32) for shape in [(2, 3), (2, 3), (2, 3, 4), (2, 3)]]

    @pipeline_def(batch_size=batch_size, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    for _ in range(len(data) // batch_size):
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


@raises(RuntimeError, "All samples in the dataset must have the same data type")
@params(1, 2)
def test_raise_different_dtype_without_arg(batch_size):
    data = [
        np.empty((2, 3), dtype=dtype) for dtype in [np.float32, np.float32, np.int32, np.float32]
    ]

    @pipeline_def(batch_size=batch_size, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=box_source(data), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    for _ in range(len(data) // batch_size):
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
    assert np.array_equal(np.array(a), np.array(b))


@raises(RuntimeError, "Expected data numpy size ")
@params(True, False)
def test_malformed_data(truncate):
    def malformed_source():
        data = np.empty(100, dtype=np.uint8)
        buff = io.BytesIO()
        np.save(buff, data, allow_pickle=False)
        buff.seek(0)
        raw = np.frombuffer(buff.read(), dtype=np.uint8)
        if truncate:
            raw = raw[:-100]  # Truncate the data
        else:
            raw = np.concatenate((raw, np.empty(100, dtype=np.uint8)))  # Add extra data
        yield (raw,)

    @pipeline_def(batch_size=1, num_threads=1)
    def pipe():
        encoded_npy = fn.external_source(
            source=malformed_source(), num_outputs=1, batch=False, ndim=1, dtype=DALIDataType.UINT8
        )
        decoded = fn.decoders.numpy(encoded_npy)
        return decoded

    p = pipe()
    p.build()
    p.run()
