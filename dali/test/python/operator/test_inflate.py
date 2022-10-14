# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import cupy as cp
import lz4.block
import os

from nvidia.dali import pipeline_def, fn
import nvidia.dali.types as types
# from types import DALIDataType

from nose_utils import assert_raises
from test_utils import dali_type_to_np, get_dali_extra_path, np_type_to_dali

# test_data_root = get_dali_extra_path()
# caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
# test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')


def sample_to_lz4(sample):
    deflated_buf = lz4.block.compress(sample, store_size=False)
    return np.frombuffer(deflated_buf, dtype=np.uint8)


def _test_sample_inflate(batch_size, np_dtype, seed):
    epoch_size = 10 * batch_size
    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(epoch_size)
    dtype = np_type_to_dali(np_dtype)

    def gen_iteration_sizes():
        num_yielded_samples = 0
        while num_yielded_samples < epoch_size:
            iteration_size = np.int32(np.floor(rng.uniform(1, batch_size + 1)))
            iteration_size = min(iteration_size, epoch_size - num_yielded_samples)
            yield iteration_size
            num_yielded_samples += iteration_size

    iteration_sizes = list(gen_iteration_sizes())
    assert sum(iteration_sizes) == epoch_size

    def source():
        num_yielded_samples = 0
        for iteration_size in iteration_sizes:
            sample_sizes = [permutation[num_yielded_samples + i] for i in range(iteration_size)]
            num_yielded_samples += iteration_size

            def sample(sample_size):
                start = (sample_size - 1) * sample_size // 2
                sample = np.array([start + i for i in range(sample_size)], dtype=np_dtype)
                return sample, sample_to_lz4(sample)

            samples, deflated = list(zip(*[sample(sample_size) for sample_size in sample_sizes]))
            yield list(samples), list(deflated), np.array(sample_sizes, dtype=np.int32)

    @pipeline_def
    def pipeline():
        sample, deflated, shape = fn.external_source(source=source, batch=True, num_outputs=3)
        inflated = fn.experimental.inflate(deflated.gpu(), shape=shape, dtype=dtype)
        return inflated, sample

    pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    pipe.build()
    for iter_size in iteration_sizes:
        inflated, baseline = pipe.run()
        inflated = [np.array(sample) for sample in inflated.as_cpu()]
        baseline = [np.array(sample) for sample in baseline]
        assert iter_size == len(inflated) == len(baseline)
        for inflated_sample, baseline_sample in zip(inflated, baseline):
            np.testing.assert_array_equal(inflated_sample, baseline_sample)


def test_sample_inflate():
    seed = 42
    for batch_size in [1, 8, 64, 256, 348]:
        for dtype in [np.uint8, np.int8, np.uint16, np.int32, np.float32, np.float16]:
            yield _test_sample_inflate, batch_size, dtype, seed
            seed += 1


def _test_scalar_shape(dtype, shape):

    def sample_source(sample_info):
        sample_size = np.prod(shape)
        x = sample_info.idx_in_epoch + 1
        sample = np.array([i * x for i in range(sample_size)], dtype=dtype).reshape(shape)
        return sample

    def deflated_source(sample_info):
        sample = sample_source(sample_info)
        return cp.array(sample_to_lz4(sample))

    @pipeline_def
    def pipeline():
        baseline = fn.external_source(source=sample_source, batch=False)
        deflated = fn.external_source(source=deflated_source, batch=False, device="gpu")
        inflated = fn.experimental.inflate(deflated, shape=shape, dtype=np_type_to_dali(dtype))
        return inflated, baseline

    pipe = pipeline(batch_size=16, num_threads=4, device_id=0)
    pipe.build()
    for _ in range(4):
        inflated, baseline = pipe.run()
        inflated = [np.array(sample) for sample in inflated.as_cpu()]
        baseline = [np.array(sample) for sample in baseline]
        len(inflated) == len(baseline)
        for inflated_sample, baseline_sample in zip(inflated, baseline):
            np.testing.assert_array_equal(inflated_sample, baseline_sample)


def test_scalar_shape():
    largest_prime_smaller_than_2_to_16 = 65521
    prime_larger_than_2_to_16 = 262147
    for shape in [
            largest_prime_smaller_than_2_to_16, prime_larger_than_2_to_16, [3, 5, 7],
            np.array([31, 101, 17], dtype=np.int32), [4, 8, 16, 2], [100, 10]
    ]:
        for dtype in [np.uint8, np.float32, np.uint16]:
            yield _test_scalar_shape, dtype, shape


# test offsets
# test offsets and sizes
# test failures