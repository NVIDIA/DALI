# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._utils import external_source_impl
from nvidia.dali import tensors, pipeline_def
import nvidia.dali.fn as fn
from nose_utils import raises, assert_equals, attr
import numpy as np


def passes_assert(callback, sample):
    assert_equals(callback(sample), True)


def converts(callback, sample, baseline):
    np.testing.assert_array_equal(callback(sample), baseline)


test_array = np.array([[42, 42], [42, 42]], dtype=np.uint8)


def run_checks(samples_allowed, batches_allowed, samples_disallowed, batches_disallowed):
    for sample, baseline in samples_allowed:
        yield passes_assert, external_source_impl.assert_cpu_sample_data_type, sample
        yield converts, external_source_impl.sample_to_numpy, sample, baseline
    for sample, baseline in samples_allowed + batches_allowed:
        yield passes_assert, external_source_impl.assert_cpu_batch_data_type, sample
        yield converts, external_source_impl.batch_to_numpy, sample, baseline
    for sample in samples_disallowed:
        yield raises(TypeError, "Unsupported callback return type.")(
            external_source_impl.assert_cpu_sample_data_type
        ), sample
    for sample in samples_disallowed + batches_disallowed:
        yield raises(TypeError, "Unsupported callback return type")(
            external_source_impl.assert_cpu_batch_data_type
        ), sample


def non_uniform_tl():
    def get_samples():
        return [np.array([42, 42]), np.array([1, 2, 3])]

    @pipeline_def(batch_size=2, num_threads=4, device_id=0)
    def pipe():
        return fn.external_source(source=get_samples)

    p = pipe()
    return p.run()[0]


def test_regular_containers():
    samples_cpu = [(test_array, test_array), (tensors.TensorCPU(test_array), test_array)]
    batches_cpu = [
        ([test_array], [test_array]),
        ([test_array] * 4, [test_array] * 4),
        ([tensors.TensorCPU(test_array)], [test_array]),
        ([tensors.TensorCPU(test_array)] * 4, [test_array] * 4),
        (tensors.TensorListCPU(test_array), test_array),
    ]
    yield from run_checks(samples_cpu, batches_cpu, [], [])


def test_non_uniform_batch():
    batches_disallowed = [[test_array, np.array([[42, 42]], dtype=np.uint8)], non_uniform_tl()]
    for b in batches_disallowed:
        yield raises(ValueError, "Uniform input is required (batch of tensors of equal shapes)")(
            external_source_impl.batch_to_numpy
        ), b


@attr("pytorch")
def test_pytorch_containers():
    import torch

    samples_cpu = [
        (torch.tensor(test_array), test_array),
    ]
    batches_cpu = [
        ([torch.tensor(test_array)], [test_array]),
        ([torch.tensor(test_array)] * 4, [test_array] * 4),
    ]
    disallowed_samples = [
        torch.tensor(test_array).cuda(),
    ]
    yield from run_checks(samples_cpu, batches_cpu, disallowed_samples, [])


@attr("mxnet")
def test_mxnet_containers():
    import mxnet as mx

    samples_cpu = [
        (mx.nd.array(test_array), test_array),
    ]
    batches_cpu = [
        ([mx.nd.array(test_array)], [test_array]),
        ([mx.nd.array(test_array)] * 4, [test_array] * 4),
    ]
    disallowed_samples = [mx.nd.array(test_array, ctx=mx.gpu(0))]
    yield from run_checks(samples_cpu, batches_cpu, disallowed_samples, [])


@attr("cupy")
def test_cupy_containers():
    import cupy as cp

    test_array = cp.array([[42, 42], [42, 42]], dtype=cp.uint8)
    disallowed_samples = [test_array, tensors.TensorGPU(test_array)]
    disallowed_batches = [tensors.TensorListGPU(test_array)]
    yield from run_checks([], [], disallowed_samples, disallowed_batches)
