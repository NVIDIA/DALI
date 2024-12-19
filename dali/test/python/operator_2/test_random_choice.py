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

from nvidia.dali import fn, pipeline_def, types

import numpy as np
import scipy.stats as st
from nose2.tools import params
from itertools import product, chain

from nose_utils import assert_raises

rng = np.random.default_rng(seed=12345)


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
    # counts are skewed. Next we flatten them so we can use the test.
    reduced_sample = np.sum(sample, axis=tuple(range(len(size), sample.ndim)))
    expected = np.sum(expected, axis=tuple(range(len(size), sample.ndim)))

    sample_values, sample_counts = np.unique(reduced_sample, return_counts=True)
    expected_values, expected_counts = np.unique(expected, return_counts=True)
    assert (
        sample_values == expected_values
    ).all(), "Bucketing is different between pipeline output and expected numpy output."

    stat = st.chisquare(
        sample_counts / np.sum(sample_counts), expected_counts / np.sum(expected_counts)
    )
    assert stat.pvalue >= 0.01, f"{stat} - distributions do not match for {idx}."


@params(
    *chain(
        product(
            ["scalar", "0d"],
            [()],
            [True, False],
            [None, (), (1000,), (500, 4)],
            [False],
        ),
        product(
            ["scalar"],
            [()],
            [True],
            [(), (1000,), (500, 4)],
            [True],
        ),
        product(
            ["nd"],
            [(2,), (2, 2)],
            [True, False],
            [None, (), (1000,), (500, 4)],
            [True, False],
        ),
    )
)
def test_choice_dist(kind, elem_shape, use_p, output_shape, shape_like):
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
        if not shape_like:
            choice = fn.random.choice(a, p=p if use_p else None, shape=output_shape)
        else:
            choice = fn.random.choice(a, np.full(output_shape, 42), p=p if use_p else None)
        return choice, a, p

    tuple_output_shape = output_shape if output_shape is not None else ()

    pipe = choice_pipe()
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


def test_choice_0_prob():
    @pipeline_def(batch_size=2, device_id=0, num_threads=4, seed=1234)
    def choice_pipe():
        return fn.random.choice(2, p=[0.0, 1.0], shape=10)

    pipe = choice_pipe()
    for _ in range(3):
        (out,) = pipe.run()
        for i in range(2):
            assert (out[i] == np.full(10, 1)).all(), "Expected all outputs to be 1."


@params(
    (1, np.array(0), 1.0),
    (1, np.array(0), None),
    (np.array([7]), np.array(7), 1.0),
    (np.array([7]), np.array(7), None),
)
def test_choice_1_elem(input, expected_output, p):
    @pipeline_def(batch_size=2, device_id=0, num_threads=4, seed=1234)
    def choice_pipe():
        return fn.random.choice(input, p=p)

    pipe = choice_pipe()
    for _ in range(3):
        (out,) = pipe.run()
        for i in range(2):
            assert (out[i] == expected_output).all(), f"Expected {expected_output}, got {out[i]}."


def test_choice_64_bit_type():
    @pipeline_def(batch_size=2, device_id=0, num_threads=4, seed=1234)
    def choice_pipe():
        a = fn.external_source(lambda: np.array(1 << 40, dtype=np.int64), batch=False)
        return fn.random.choice(a)

    pipe = choice_pipe()
    for _ in range(3):
        (out,) = pipe.run()
        assert out.dtype == types.INT64, f"Expected {types.INT64}, got {out.dtype}."
        for i in range(2):
            assert (0 <= np.array(out[i]) < (1 << 40)).all(), f"Output out of range: {out[i]}."


@params(
    (np.arange(3), "N", "N", 10),
    (np.arange(3), "N", "", None),
    (np.full((2, 3), 4), "NS", "S", None),
    (np.full((2, 3), 4), "NS", "NS", 5),
    (np.full((2, 3), 4), "NS", "", (1, 2)),
)
def test_layout(input, input_layout, expected_layout, shape):
    @pipeline_def(batch_size=2, device_id=0, num_threads=4, seed=1234)
    def choice_pipe():
        a = fn.external_source(lambda: input, batch=False, layout=input_layout)
        return fn.random.choice(a, shape=shape)

    pipe = choice_pipe()
    for _ in range(3):
        (out,) = pipe.run()
        assert (
            out.layout() == expected_layout
        ), f'Expected layout "{expected_layout}", got {out.layout()}.'


@params(
    (
        (1,),
        {"p": 1.5},
        "Probabilities must be in range *, but got: 1.5 for sample: 0 at index 0.",
    ),
    (
        (2,),
        {"p": 0.25},
        "Sum of probabilities must be 1.0, but got 0.5 for sample: 0.",
    ),
    (
        (-5,),
        {},
        "Expected positive number of elements for sampling, got: -5 for sample: 0.",
    ),
    (
        (0,),
        {},
        "Expected positive number of elements for sampling, got: 0 for sample: 0.",
    ),
    (
        (5.0,),
        {},
        "Data type float is not supported for 0D inputs. Supported types are: "
        "uint8, uint16, uint32, uint64, int8, int16, int32, int64",
    ),
    (
        (types.DALIInterpType.INTERP_CUBIC,),
        {},
        "Data type DALIInterpType is not supported for 0D inputs. Supported types are: "
        "uint8, uint16, uint32, uint64, int8, int16, int32, int64",
    ),
    (
        (5,),
        {"p": np.array([0.25, 0.5, 0.25])},
        'Unexpected shape for argument "p". Expected {5}, but got {3}',
    ),
)
def test_choice_validation(args, kwargs, expected_error):

    with assert_raises(RuntimeError, glob=expected_error):

        @pipeline_def(batch_size=1, device_id=0, num_threads=4)
        def choice_pipe():
            values = fn.random.choice(*args, **kwargs)
            return values

        pipe = choice_pipe()
        pipe.run()


def test_enum_choice():
    batch_size = 8

    interps_to_sample = [types.DALIInterpType.INTERP_LINEAR, types.DALIInterpType.INTERP_CUBIC]

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def choice_pipeline():
        interp = fn.random.choice(interps_to_sample, shape=[100])
        interp_as_int = fn.cast(interp, dtype=types.INT32)
        imgs = fn.resize(
            fn.random.uniform(range=[0, 255], dtype=types.UINT8, shape=(100, 100, 3)),
            size=(25, 25),
            interp_type=interp[0],
        )
        return interp, interp_as_int, imgs

    pipe = choice_pipeline()
    (interp, interp_as_int, imgs) = pipe.run()
    assert interp.dtype == types.DALIDataType.INTERP_TYPE
    for i in range(batch_size):
        check_sample(
            np.array(interp_as_int[i]),
            size=(100,),
            a=np.array([v.value for v in interps_to_sample]),
            p=None,
            idx=i,
        )
