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

from nvidia.dali import fn, pipeline_def

import numpy as np
import scipy.stats as st
from nose2.tools import cartesian_params


rng = np.random.default_rng(seed=12345)


def check_sample(sample, a, p, idx):
    flat_sample = sample.flatten()
    # if a.shape == ():
    #     expected_set = set(np.arange(a))
    # else:
    #     expected_set = set(a)
    # got = set(flat_sample)
    # assert (
    #     got == expected_set
    # ), f"Sampled values don't match expected set, got {got}, expected {expected_set}"
    expected = rng.choice(a, size=flat_sample.shape, p=p).flatten()
    result = st.kstest(flat_sample, expected)
    assert (
        result[1] > 0.05
    ), f"Sample is not identical to expected distribution, {result}, for {idx}"


@cartesian_params(
    [True, False],
    [True, False],
    [
        None,
        (3,),
        (4, 2),
    ],
)
def test_0d_elements(scalar_input, use_p, shape):
    sampling_offset = 5  # offset compared to the index in the batch
    batch_size = 8
    n_iters = 1000

    def inp(sample_info):
        a = sample_info.idx_in_batch + sampling_offset
        p = np.zeros((a), dtype=np.float32)
        for i in range(a):
            p[i] = (i + 1) / (a * (a + 1) / 2)
        if scalar_input:
            return np.array(a), p
        else:
            return np.arange(sampling_offset, a + sampling_offset), p

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, seed=12345)
    def choice_pipe():
        a, p = fn.external_source(inp, batch=False, num_outputs=2)
        choice = fn.random.choice(a, p=p if use_p else None, shape=shape)
        return choice, a, p

    pipe = choice_pipe()
    pipe.build()
    choices = [[] for _ in range(batch_size)]
    for _ in range(n_iters):
        ch, a, p = pipe.run()
        for i in range(batch_size):
            assert tuple(ch[i].shape()) == (shape if shape is not None else ())
            # Extract and accumulate values in samplewise fashion
            choices[i].append(np.array(ch[i]))
    for i in range(batch_size):
        choices[i] = np.stack(choices[i])
        check_sample(
            choices[i],
            a=np.array(a[i]),
            p=np.array(p[i]) if use_p else None,
            idx=i,
        )


@cartesian_params(
    [
        (2,),
        (4, 2),
    ],
    [True, False],
    [
        None,
        (3,),
        (4, 2),
    ],
)
def test_nd_elements(elem_shape, use_p, shape):
    sampling_offset = 5  # offset compared to the index in the batch
    batch_size = 8
    n_iters = 1000

    def inp(sample_info):
        elem_cout = sample_info.idx_in_batch + sampling_offset
        p = np.zeros((elem_cout), dtype=np.float32)
        for i in range(elem_cout):
            p[i] = (i + 1) / (elem_cout * (elem_cout + 1) / 2)
        return np.stack([np.full(elem_shape, i) for i in range(elem_cout)]), p

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, seed=12345)
    def choice_pipe():
        a, p = fn.external_source(inp, batch=False, num_outputs=2)
        choice = fn.random.choice(a, p=p if use_p else None, shape=shape)
        return choice, a, p

    pipe = choice_pipe()
    pipe.build()
    choices = [[] for _ in range(batch_size)]
    for _ in range(n_iters):
        ch, a, p = pipe.run()
        for i in range(batch_size):
            assert (
                tuple(ch[i].shape()) == (shape if shape is not None else ()) + elem_shape
            ), f"{ch[i].shape()}!= {(shape if shape is not None else ())} + {elem_shape}"
            # Extract and accumulate values in samplewise fashion
            choices[i].append(np.array(ch[i]))
    for i in range(batch_size):
        choices[i] = np.stack(choices[i])
        check_sample(
            choices[i],
            a=np.array(a[i]),
            p=np.array(p[i]) if use_p else None,
            idx=i,
        )
