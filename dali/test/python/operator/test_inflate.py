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
import lz4.block

from nvidia.dali import pipeline_def, fn
from test_utils import np_type_to_dali


def sample_to_lz4(sample):
    deflated_buf = lz4.block.compress(sample, store_size=False)
    return np.frombuffer(deflated_buf, dtype=np.uint8)


def check_batch(inflated, baseline, batch_size, layout=None, oversized_shape=False):
    layout = layout or ""
    assert inflated.layout() == layout, (f"The batch layout '({inflated.layout()})' does "
                                         f"not match the expected layout ({layout})")
    inflated_samples = [np.array(sample) for sample in inflated.as_cpu()]
    baseline_samples = [np.array(sample) for sample in baseline]
    assert batch_size == len(inflated) == len(baseline)
    if not oversized_shape:
        for inflated_sample, baseline_sample in zip(inflated_samples, baseline_samples):
            np.testing.assert_array_equal(inflated_sample, baseline_sample)
    else:
        for inflated_sample, baseline_sample in zip(inflated_samples, baseline_samples):
            assert len(inflated_sample) == len(baseline_sample)
            for inflated_frame, baseline_frame in zip(inflated_sample, baseline_sample):
                flat_inflated = inflated_frame.reshape(-1)
                baseline_size = baseline_frame.size
                actually_inflated = flat_inflated[:baseline_size].reshape(baseline_frame.shape)
                np.testing.assert_array_equal(actually_inflated, baseline_frame)
                output_tail = flat_inflated[baseline_size:]
                assert np.all(
                    output_tail == 0), (f"Oversized output was not properly padded with 0s. "
                                        f"Tail size {len(output_tail)}, the tail {output_tail}")


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
        check_batch(inflated, baseline, iter_size)


def test_sample_inflate():
    seed = 42
    for batch_size in [1, 8, 64, 256, 348]:
        for dtype in [np.uint8, np.int8, np.uint16, np.int32, np.float32, np.float16]:
            yield _test_sample_inflate, batch_size, dtype, seed
            seed += 1


def _test_scalar_shape(dtype, shape, layout):

    def sample_source(sample_info):
        sample_size = np.prod(shape)
        x = sample_info.idx_in_epoch + 1
        sample = np.array([i * x for i in range(sample_size)], dtype=dtype).reshape(shape)
        return sample

    def deflated_source(sample_info):
        sample = sample_source(sample_info)
        return np.array(sample_to_lz4(sample))

    @pipeline_def
    def pipeline():
        baseline = fn.external_source(source=sample_source, batch=False)
        deflated = fn.external_source(source=deflated_source, batch=False, device="gpu")
        inflated = fn.experimental.inflate(deflated, shape=shape, dtype=np_type_to_dali(dtype),
                                           layout=layout)
        return inflated, baseline

    batch_size = 16
    pipe = pipeline(batch_size=batch_size, num_threads=8, device_id=0)
    pipe.build()
    for _ in range(4):
        inflated, baseline = pipe.run()
        check_batch(inflated, baseline, batch_size, layout)


def test_scalar_shape():
    largest_prime_smaller_than_2_to_16 = 65521
    prime_larger_than_2_to_16 = 262147
    for shape, layout in [(largest_prime_smaller_than_2_to_16, "X"),
                          (largest_prime_smaller_than_2_to_16, None),
                          (prime_larger_than_2_to_16, "Y"), ([3, 5, 7], "ABC"), ([3, 5, 7], ""),
                          ([13, 15, 7], None), (np.array([31, 101, 17], dtype=np.int32), "DEF"),
                          ([4, 8, 16, 2], "FGNH"), ([100, 10], "WW")]:
        for dtype in [np.uint8, np.float32, np.uint16]:
            yield _test_scalar_shape, dtype, shape, layout


def seq_source(rng, ndim, dtype, mode, permute, oversized_shape):

    def uniform(shape):
        return dtype(rng.uniform(-2**31, 2**31 - 1, shape))

    def std(shape):
        return dtype(128 * rng.standard_normal(shape) + 3)

    def smaller_std(shape):
        return dtype(16 * rng.standard_normal(shape))

    def inflate_shape(shape):
        multiplier = rng.uniform(1, 2, ndim)
        return np.int32(shape * multiplier)

    def inner():
        max_extent_size = 64 if ndim >= 3 else 128
        distrs = [uniform, std, smaller_std]
        distrs = rng.permutation(distrs)
        num_chunks = np.int32(rng.uniform(1, 32))
        shape = np.int32(rng.uniform(0, max_extent_size, ndim))
        sample = np.array([(distrs[i % len(distrs)])(shape) for i in range(num_chunks)],
                          dtype=dtype)
        chunks = [sample_to_lz4(chunk) for chunk in sample]
        sizes = [len(chunk) for chunk in chunks]
        offsets = np.int32(np.cumsum([0] + sizes[:-1]))
        sizes = np.array(sizes, dtype=np.int32)
        deflated = np.concatenate(chunks)
        reported_shape = shape if not oversized_shape else inflate_shape(shape)
        if permute:
            assert mode == "offset_and_size"
            perm = rng.permutation(num_chunks)
            sample = sample[perm]
            offsets = offsets[perm]
            sizes = sizes[perm]
        if mode == "offset_only":
            return sample, deflated, reported_shape, offsets
        elif mode == "size_only":
            return sample, deflated, reported_shape, sizes
        else:
            assert mode == "offset_and_size"
            return sample, deflated, reported_shape, offsets, sizes

    return inner


def _test_chunks(seed, batch_size, ndim, dtype, layout, mode, permute, oversized_shape):
    rng = np.random.default_rng(seed=seed)
    source = seq_source(rng, ndim, dtype, mode, permute, oversized_shape)

    @pipeline_def
    def pipeline():
        baseline, deflated, reported_shape, *rest = fn.external_source(
            source=source, batch=False, num_outputs=5 if mode == "offset_and_size" else 4)
        if mode == "offset_only":
            offsets, = rest
            sizes = None
        elif mode == "size_only":
            sizes, = rest
            offsets = None
        else:
            offsets, sizes = rest
        inflated = fn.experimental.inflate(deflated.gpu(), shape=reported_shape,
                                           dtype=np_type_to_dali(dtype), chunks_offsets=offsets,
                                           chunks_sizes=sizes, layout=layout)
        return inflated, baseline

    pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    pipe.build()
    if layout:
        layout = "F" + layout
    for _ in range(4):
        inflated, baseline = pipe.run()
        check_batch(inflated, baseline, batch_size, layout, oversized_shape=oversized_shape)


def test_chunks():
    seed = 42
    batch_sizes = [1, 9, 31]
    for dtype in [np.uint8, np.int16, np.float32]:
        for ndim, layout in [(0, None), (1, None), (2, "XY"), (2, None), (3, "ABC"), (3, "")]:
            for mode, permute in [("offset_only", False), ("size_only", False),
                                  ("offset_and_size", False), ("offset_and_size", True)]:
                batch_size = batch_sizes[seed % len(batch_sizes)]
                oversized_shape = ndim > 0 and seed % 2 == 1
                yield _test_chunks, seed, batch_size, ndim, dtype, layout, mode, \
                    permute, oversized_shape
                seed += 1
