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
from nose2.tools import cartesian_params, params
from itertools import product, chain


rng = np.random.default_rng(seed=12345)


def get_counts(sample):
    unique, counts = np.unique(sample, return_counts=True)
    counts = dict(zip(unique, counts))
    return counts


def check_sample(sample, size, a, p, idx):
    flat_sample = sample.flatten()
    if a.shape == ():
        expected_set = set(np.arange(a))
    else:
        expected_set = set(a.flatten())
    got = set(flat_sample)
    assert (
        got == expected_set
    ), f"Sampled values don't match expected set, got {got}, expected {expected_set}"

    expected = rng.choice(a, size=size, p=p)
    # In N-D case, we reduce over the sample dimensions as otherwise the distribution
    # counts are skewed and kstest fails. As we use samples of unique repeated numbers,
    # this is ok. Next we flatten them so we can use the test.
    reduced_sample = np.sum(sample, axis=tuple(range(len(size), sample.ndim))).flatten()
    expected = np.sum(expected, axis=tuple(range(len(size), sample.ndim))).flatten()
    print(f"{get_counts(reduced_sample)=}\n{get_counts(expected)=}")

    result = st.kstest(reduced_sample, expected)
    assert (
        result[1] > 0.05
    ), f"Sample is not identical to expected distribution, {result}, for {idx}"


@params(
    *chain(
        product(
            ["scalar", "0d"],
            [()],
            [True, False],
            [None, (), (1000,), (500, 4)],
        ),
        product(
            ["nd"],
            [(2,), (2, 2)],
            [True, False],
            [None, (), (1000,), (500, 4)],
        ),
    )
)
def test_choice_dist(kind, elem_shape, use_p, output_shape):
    """Test if fn.random.choice distribution is matching the np.random.Generator.choice.

    Parameters
    ----------
    kind : str
        What kind of `a` parameter to test: "scalar" - just a scalar value, "0d" - list of scalars,
        or "nd" - "list" of n-d elements.
    elem_shape : tuple of int
        Shape of the elements sampled by the distribution, non-empty tuple allowed only for "nd"
        case.
    use_p : bool
        Pass the p parameter or use uniform distribution.
    output_shape : tuple
        Parameter requesting the shape of output, the resulting shape will be
        `output_shape + elem_shape`.
    """

    if kind in {"scalar", "0d"}:
        assert elem_shape == (), "elem_shape must be empty tuple for scalar and 0d case"
    sampling_offset = 5  # offset compared to the index in the batch
    batch_size = 8
    if output_shape is None or output_shape == ():
        n_iters = 1000
    else:
        n_iters = 5

    def get_p(n):
        """
        Return probabilities for n elements, i-th element is proportional to its index in batch.
        """
        p = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            p[i] = (i + 1) / (n * (n + 1) / 2)
        return p

    def inp(sample_info):
        n = sample_info.idx_in_batch + sampling_offset
        p = get_p(n)
        if kind == "scalar":
            return np.array(n), p
        elif kind == "0d":
            return np.arange(sampling_offset, n + sampling_offset), p
        else:
            # nd - repeat the same scalar value with the elem_shape for given sample
            return np.stack([np.full(elem_shape, i) for i in range(n)]), p

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, seed=1234)
    def choice_pipe():
        a, p = fn.external_source(inp, batch=False, num_outputs=2)
        choice = fn.random.choice(a, p=p if use_p else None, shape=output_shape)
        return choice, a, p

    tuple_output_shape = output_shape if output_shape is not None else ()

    pipe = choice_pipe()
    pipe.build()
    choices = [[] for _ in range(batch_size)]
    for _ in range(n_iters):
        ch, a, p = pipe.run()
        for i in range(batch_size):
            assert tuple(ch[i].shape()) == tuple_output_shape + elem_shape
            # Extract and accumulate values in samplewise fashion
            choices[i].append(np.array(ch[i]))
    for i in range(batch_size):
        choices[i] = np.stack(choices[i])
        check_sample(
            choices[i],
            size=(n_iters,) + tuple_output_shape,
            a=np.array(a[i]),
            p=np.array(p[i]) if use_p else None,
            idx=i,
        )
