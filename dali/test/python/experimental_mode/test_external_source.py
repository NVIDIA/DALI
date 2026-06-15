# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools

import numpy as np
import nvidia.dali.backend as _backend
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nose_utils import SkipTest, assert_raises


def _samples(n=3):
    return [np.full((2, 2), i) for i in range(n)]


def test_empty_data():
    es = ndd.ExternalSource(lambda: ())
    np.testing.assert_array_equal(es(), ())


def test_callable_source():
    counter = itertools.count()
    es = ndd.ExternalSource(lambda: np.full((2, 2), next(counter)))
    for value in range(3):
        np.testing.assert_array_equal(es(), np.full((2, 2), value))


def test_iterable_source():
    samples = _samples()
    es = ndd.ExternalSource(samples)
    for expected in samples:
        np.testing.assert_array_equal(es(), expected)


def test_generator_source():
    def gen():
        yield from samples

    samples = _samples()
    es = ndd.ExternalSource(gen)
    for expected in samples:
        np.testing.assert_array_equal(es(), expected)


def test_sample_output():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    es = ndd.ExternalSource(lambda: data)
    result = es()
    assert isinstance(result, ndd.Tensor)
    np.testing.assert_array_equal(result, data)


def test_batch_broadcast():
    batch_size = 4
    data = np.array([1, 2, 3])
    es = ndd.ExternalSource(lambda: data)
    result = es(batch_size=batch_size)
    assert isinstance(result, ndd.Batch)
    assert result.batch_size == batch_size
    np.testing.assert_array_equal(ndd.as_tensor(result), np.stack([data] * batch_size))


def test_batch_output():
    data = _samples()
    es = ndd.ExternalSource(lambda: ndd.batch(data))
    result = es()
    assert isinstance(result, ndd.Batch)
    assert result.batch_size == len(data)
    np.testing.assert_array_equal(ndd.as_tensor(result), np.stack(data))


def test_tensor_list_output():
    data = _samples()
    es = ndd.ExternalSource(lambda: _backend.TensorListCPU(np.stack(data)))
    result = es()
    assert isinstance(result, ndd.Batch)
    assert result.batch_size == len(data)
    np.testing.assert_array_equal(ndd.as_tensor(result), np.stack(data))


def test_same_batch_size():
    data = _samples()
    es = ndd.ExternalSource(lambda: ndd.batch(data))
    result = es(batch_size=len(data))
    assert isinstance(result, ndd.Batch)
    assert result.batch_size == len(data)


def test_different_batch_size():
    es = ndd.ExternalSource(lambda: ndd.batch(_samples(2)))
    with assert_raises(ValueError, glob="expected batch size 5"):
        es(batch_size=5)


def test_num_outputs():
    a = np.array([1, 2])
    b = np.array([3, 4, 5])
    es = ndd.ExternalSource(lambda: (a, b), num_outputs=2)
    out = es()
    assert isinstance(out, tuple)
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], a)
    np.testing.assert_array_equal(out[1], b)


def test_bad_num_outputs():
    es = ndd.ExternalSource(lambda: (np.zeros(2),), num_outputs=2)
    with assert_raises(ValueError, glob="expected 2 outputs"):
        es()


@params("no", False, None)
def test_cycle_disabled(cycle):
    es = ndd.ExternalSource(_samples(2), cycle=cycle)
    es()
    es()
    with assert_raises(StopIteration):
        es()


@params("quiet", True)
def test_cycle_quiet(cycle):
    samples = _samples(2)
    es = ndd.ExternalSource(samples, cycle=cycle)
    for expected in samples + samples:
        np.testing.assert_array_equal(es(), expected)


def test_cycle_raise():
    samples = _samples(2)
    es = ndd.ExternalSource(samples, cycle="raise")
    es()
    es()
    with assert_raises(StopIteration):
        es()
    # The iteration restarts on the subsequent call.
    np.testing.assert_array_equal(es(), samples[0])


def test_cycle_callable_rejected():
    with assert_raises(ValueError, glob="cycle"):
        ndd.ExternalSource(lambda: np.zeros(2), cycle="quiet")


def test_callable_argument_rejected():
    with assert_raises(ValueError, glob="no parameters"):
        ndd.ExternalSource(lambda sample_info: np.zeros(2))


def test_source_layouts():
    es = ndd.ExternalSource(
        lambda: (np.zeros((4, 4, 3)), np.zeros((4, 4))),
        num_outputs=2,
        layout=["HWC", "HW"],
    )
    a, b = es()
    assert a.layout == "HWC"
    assert b.layout == "HW"


def test_layouts_bad_number():
    with assert_raises(ValueError, glob="expected a sequence of size 2"):
        ndd.ExternalSource(
            lambda: (np.zeros((4, 4, 3)), np.zeros((4, 4))),
            num_outputs=2,
            layout=["HWC", "HW", "CHW"],
        )


def test_source_dtypes():
    es = ndd.ExternalSource(
        lambda: (np.zeros(3), np.zeros(3)),
        num_outputs=2,
        dtype=[ndd.float32, ndd.int32],
    )
    a, b = es()
    assert a.dtype == ndd.float32
    assert b.dtype == ndd.int32


def test_dtypes_bad_number():
    with assert_raises(ValueError, glob="expected a sequence of size 3"):
        ndd.ExternalSource(
            lambda: (np.zeros(3), np.zeros(3), np.zeros(3)),
            num_outputs=3,
            dtype=[ndd.float32, ndd.int32],
        )


@params("cpu", "gpu")
def test_device(device_type):
    if device_type == "gpu" and _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU device needed for the test")
    data = np.array([1, 2, 3])
    es = ndd.ExternalSource(lambda: data, device=device_type)
    result = es()
    assert result.device == ndd.Device(device_type)
    np.testing.assert_array_equal(result.cpu(), data)
