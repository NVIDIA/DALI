# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import numpy as np
import scipy as sp

from nose2.tools import params

from nvidia.dali import fn, pipeline_def, types
from nose_utils import assert_raises


@params(
    (0, 1, 1, np.float32),
    (1, 1, 1, np.float64),
    (2, 10, 10, np.float32),
    (3, 100, 100, np.float32),
    (4, 1, 3, np.float32),
    (5, 7, 4, np.float32),
    (6, 99, 105, np.float32),
    (7, 13, 99, np.float32),
    (8, 1011, 13, np.float32),
    (9, 1011, 13, np.float64),
    (10, 0.81, 100, np.float32),
    (11, 100, 0.61, np.float32),
    (12, 1, 1e7, np.float32),
    (13, 1e7, 1, np.float64),
    (14, 0.2, 0.5, np.float32),
    (15, 0.112, 0.417, np.float32),
    (16, 0.713, 0.621, np.float64),
    (17, 0.003, 0.002, np.float32),
    (18, 0.00003, 0.00002, np.float64),
    (19, 1e35, 0.8, np.float64),
    (20, 11 * 1e-25, 17 * 1e-31, np.float32),
    (21, 13 * 1e-7, 123 * 1e-7, np.float32),
    (22, 3 * 1e-37, 2 * 1e-35, np.float32),
    (23, 3 * 1e-37, 5 * 1e-37, np.float32),
    (24, 1e-44, 1e-43, np.float32),
    (25, 1e-43, 1e-44, np.float64),
    (26, 1000 * 1e-15, 1712 * 1e-15, np.float64),
    (27, 3 * 1e-35, 2 * 1e-37, np.float64),
    (28, 2 * 1e-36, 3 * 1e-37, np.float64),
    (29, 12345, 10000, np.float32),
    (30, 1e15, 123 * 1e15, np.float64),
)
def test_beta_distribution(case_idx, alpha, beta, dtype):

    shape = (10, 5, 2, 5, 5, 2, 2, 10)
    size = math.prod(shape)
    assert size == 1e5
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    seed = 42 + case_idx
    batch_size = 4
    dali_dtype = types.DALIDataType.FLOAT if dtype == np.float32 else types.DALIDataType.FLOAT64

    rns = np.random.RandomState(seed=seed)
    sp_beta_sampler = sp.stats.beta(alpha, beta).rvs

    @pipeline_def(
        batch_size=batch_size,
        device_id=0,
        num_threads=4,
        seed=seed,
        prefetch_queue_depth=1,
    )
    def pipeline():
        shape_provider = fn.full(1, shape=shape)
        return fn.random.beta(
            shape_provider,
            alpha=alpha,
            beta=beta,
            dtype=dali_dtype,
        )

    p = pipeline()
    batch = [np.array(s) for s in p.run()[0]]
    assert all(s.dtype == dtype for s in batch)
    assert all(s.shape == shape for s in batch)
    batch = [s.reshape(-1) for s in batch]
    refs = [
        sp_beta_sampler(
            size=(size,),
            random_state=rns,
        ).astype(dtype)
        for _ in range(batch_size)
    ]
    assert len(batch) == len(refs)
    pvs = [sp.stats.kstest(sample, sample_ref)[1] for sample, sample_ref in zip(batch, refs)]

    # We're running a lot of tests in total so having a random failure
    # with 0.01 threshold is quite likely. Hence, the threshold for a single
    # sample is smaller (1e-4), but we perform a couple of tests for given params
    # and additionally validate the mean against higher threshold (0.05)
    # to catch consistenly underperforming cases
    assert sum(pvs) / batch_size >= 0.05, f"{alpha} {beta} {pvs}"
    assert all(pv >= 1e-4 for pv in pvs), f"{alpha} {beta} {pvs}"

    s_min, s_max = np.min(batch), np.max(batch)
    assert s_min >= 0, f"{s_min} {alpha} {beta}"
    assert s_max <= 1, f"{s_max} {alpha} {beta}"
    actual_mean = np.mean(batch)
    expected_mean = dtype(alpha) / (dtype(alpha) + dtype(beta))
    # sqrt(variance)
    o = np.exp(
        0.5 * (np.log(alpha) + np.log(beta) - 2 * np.log(alpha + beta) - np.log(alpha + beta + 1))
    )
    df = np.abs(actual_mean - expected_mean).astype(np.float32)
    assert df <= min(o, 0.01), f"{expected_mean}, {actual_mean}, {o}"


@params(
    (0, 1, 200, np.float32),
    (1, 200, 400, np.float32),
    (2, 0.1, 2, np.float32),
    (3, 0.1, 1.9, np.float64),
    (4, 1e18, 1e19, np.float64),
)
def test_beta_distribution_tensor_input(case_idx, min_param, max_param, dtype):

    shape = (4, 25, 10, 10, 10)
    size = math.prod(shape)
    seed = 142 + case_idx
    batch_size = 4
    rng = np.random.default_rng(seed=seed)
    dali_dtype = types.DALIDataType.FLOAT if dtype == np.float32 else types.DALIDataType.FLOAT64
    num_attempts = 3

    @pipeline_def(
        batch_size=batch_size,
        device_id=0,
        num_threads=8,
        seed=seed,
        prefetch_queue_depth=2,
    )
    def pipeline():
        var_batch = fn.external_source(
            lambda i: np.full((batch_size, 1), i) if i == 0 else np.full((batch_size // 2, 1), i),
            batch=True,
        )
        a = fn.random.uniform(range=[min_param, max_param])
        b = fn.random.uniform(range=[min_param, max_param])
        return (
            a,
            b,
            fn.random.beta(
                alpha=a,
                beta=b,
                shape=(num_attempts,) + shape,
                dtype=dali_dtype,
            ),
            var_batch,
        )

    p = pipeline()
    for i in range(2):
        alps, bs, samples, _ = p.run()
        alps = [np.array(a) for a in alps]
        bs = [np.array(b) for b in bs]
        samples = [np.array(s) for s in samples]
        assert len(samples) == batch_size // (2**i)
        assert all(s.dtype == dtype for s in samples)
        assert all(s.shape == (num_attempts,) + shape for s in samples)
        samples = [s.reshape(num_attempts, -1) for s in samples]
        refs = [rng.beta(a, b, size=(num_attempts, size)).astype(dtype) for a, b in zip(alps, bs)]
        assert len(samples) == len(refs), f"{len(samples)} vs {len(refs)}"
        assert len(alps) == len(bs) == len(samples)
        for a, b, sample, sample_ref in zip(alps, bs, samples, refs):
            assert sample.shape == sample_ref.shape, f"{sample.shape} vs {sample_ref.shape}"
            pvs = [sp.stats.kstest(s, sr)[1] for s, sr in zip(sample, sample_ref)]
            assert sum(pvs) / num_attempts >= 0.05, f"{a} {b} {pvs}"
            assert all(pv >= 1e-4 for pv in pvs), f"{a} {b} {pvs}"

            s_min, s_max = np.min(sample), np.max(sample)
            assert s_min >= 0, f"{s_min} {a} {b}"
            assert s_max <= 1, f"{s_max} {a} {b}"
            actual_mean = np.mean(sample)
            expected_mean = dtype(a) / (dtype(a) + dtype(b))
            o = np.exp(0.5 * (np.log(a) + np.log(b) - 2 * np.log(a + b) - np.log(a + b + 1)))
            df = np.abs(actual_mean - expected_mean)
            assert df <= min(o, 0.01), f"{expected_mean}, {actual_mean}, {o}, {a}, {b}"


@params(
    (0, 1, None, "The `alpha` must be a positive float32"),
    (1, 0, None, "The `beta` must be a positive float32"),
    (-0.1, 0, None, "The `alpha` must be a positive float32"),
    (1e100, 5, None, "The `alpha` must be a positive float32"),
    (1, 1, types.DALIDataType.FLOAT16, "Data type float16 is not supported"),
    (1, 1, types.DALIDataType.INT32, "Data type int32 is not supported"),
    (np.full((2, 2), 1), 1, types.DALIDataType.FLOAT, "cannot be converted"),
    (1, np.full((1, 1), 1), types.DALIDataType.FLOAT, "cannot be converted"),
    (1, np.full((2,), 1), types.DALIDataType.FLOAT, "cannot be converted"),
)
def test_incorrect_param(a, b, dtype, msg):

    @pipeline_def(batch_size=2, device_id=0, num_threads=4, seed=111)
    def pipeline():
        return fn.random.beta(alpha=a, beta=b, dtype=dtype)

    with assert_raises(RuntimeError, glob=msg):
        p = pipeline()
        p.run()
