# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import SupportsIndex
import numpy as np
from nose2.tools import cartesian_params, params
from nose_utils import assert_raises, raises

import nvidia.dali.experimental.dynamic as ndd

ops = {
    "uniform": ndd._ops.random.Uniform,
    "normal": ndd._ops.random.Normal,
}

fn = {
    "uniform": ndd.random.uniform,
    "normal": ndd.random.normal,
}

op_args = {
    "uniform": {"range": [0.0, 1.0], "shape": [10]},
    "normal": {"mean": 0.0, "stddev": 1.0, "shape": [10]},
}


@cartesian_params(("cpu", "gpu"), (None, 3), ("ops", "fn"), ("uniform", "normal"))
def test_rng_argument(device_type, batch_size, api_type, opname):
    """Test that the rng argument works with random operators."""
    op_instance = None

    def generate(rng):
        nonlocal op_instance
        # Create operator or use functional API
        if api_type == "ops":
            if op_instance is None:
                op_instance = ops[opname](device=device_type, max_batch_size=batch_size)
            result = op_instance(batch_size=batch_size, rng=rng, **op_args[opname])
        else:
            result = fn[opname](
                batch_size=batch_size, rng=rng, device=device_type, **op_args[opname]
            )

        # Verify result type and shape
        if batch_size is not None:
            assert isinstance(result, ndd.Batch), f"Expected Batch, got {type(result)}"
            tensor = ndd.as_tensor(result, device="cpu")
            assert tensor.shape == (
                batch_size,
                10,
            ), f"Expected shape ({batch_size}, 10), got {tensor.shape}"
        else:
            assert isinstance(result, ndd.Tensor), f"Expected Tensor, got {type(result)}"
            tensor = result.cpu()
            assert tensor.shape == (10,), f"Expected shape (10,), got {tensor.shape}"
        return tensor

    rng1 = ndd.random.RNG(seed=1234)
    rng2 = ndd.random.RNG(seed=1234)
    result1_np = generate(rng1)
    result2_np = generate(rng1)
    result3_np = generate(rng2)
    assert not np.array_equal(
        result1_np, result2_np
    ), "Results should not be identical with different random state"
    assert np.array_equal(
        result1_np, result3_np
    ), "Results should be identical with same random state"


@params(("cpu",), ("gpu",))
def test_rng_seed_exclusion(device_type):
    """Test that seed argument is removed when rng is provided."""
    rng1 = ndd.random.RNG(seed=1111)
    rng2 = ndd.random.RNG(seed=2222)

    # This should work - rng should override seed (seed is an init-time argument)
    uniform_op1 = ndd._ops.random.Uniform(device=device_type, seed=42)
    uniform_op2 = ndd._ops.random.Uniform(device=device_type, seed=42)
    result1 = uniform_op1(range=[0.0, 1.0], shape=[10], rng=rng1)  # This should override the seed
    result2 = uniform_op2(range=[0.0, 1.0], shape=[10], rng=rng2)  # This should override the seed

    assert result1.shape == (10,)
    assert result2.shape == (10,)

    # expected to be different because of different random states
    # regardless of the initial seed
    assert not np.array_equal(result1.cpu(), result2.cpu())


def test_rng_clone():
    """Test that RNG.clone() creates an independent copy with the same seed."""
    # Create an RNG with a specific seed
    rng1 = ndd.random.RNG(seed=5678)
    _ = rng1()  # advance
    _ = rng1()  # advance some more

    # Clone it
    rng2 = rng1.clone()

    # set state - effectively clone
    rng3 = ndd.random.RNG(seed=12345)  # different seed
    rng3.state = rng2.state

    # Verify they are different objects
    assert rng1 is not rng2, "Clone should create a new object"

    # Verify they generate the same sequence
    for i in range(10):
        val1 = rng1()
        val2 = rng2()
        val3 = rng3()
        assert val1 == val2, f"Value {i} doesn't match: {val1} != {val2}"
        assert val1 == val3, f"Value {i} doesn't match: {val1} != {val3}"

    # Verify cloned RNG works with operators
    rng4 = ndd.random.RNG(seed=9999)
    for _ in range(10):
        _ = rng4()  # advance by 10 iterations
    rng5 = rng4.clone()

    result1 = ndd.random.uniform(range=[0.0, 1.0], shape=[10], rng=rng4)
    result2 = ndd.random.uniform(range=[0.0, 1.0], shape=[10], rng=rng5)

    # Results should be identical since clones have the same seed
    assert np.array_equal(result1, result2), "Cloned RNGs should produce identical operator results"


@raises(ValueError, glob="both")
def test_rng_init_seed_and_state_error():
    ndd.random.RNG(seed=0, state="")


def test_batch_permutation_requires_batch_size():
    with assert_raises(ValueError, glob="*batch_size*batch_permutation*"):
        ndd.batch_permutation(rng=ndd.random.RNG(seed=1234))

    batch_size = 4
    result = ndd.batch_permutation(batch_size=batch_size, rng=ndd.random.RNG(seed=1234))
    tensor = ndd.as_tensor(result, device="cpu")
    assert tensor.shape == (batch_size,)
    assert np.array_equal(np.sort(tensor), np.arange(batch_size))


def test_rng_set_seed():
    # Explicit RNG instance
    rng = ndd.random.RNG(seed=1234)
    values1 = [rng() for _ in range(5)]
    rng.seed = 1234
    values2 = [rng() for _ in range(5)]
    assert values1 == values2
    rng.seed = 5678  # Different seed should produce different values
    values3 = [rng() for _ in range(5)]
    assert values1 != values3

    # Explicit RNG instance with operators
    rng.seed = 1234
    result1 = ndd.random.uniform(range=[0.0, 1.0], shape=[10], rng=rng)
    rng.seed = 1234
    result2 = ndd.random.uniform(range=[0.0, 1.0], shape=[10], rng=rng)
    assert np.array_equal(result1, result2)
    rng.seed = 5678  # Different seed
    result3 = ndd.random.uniform(range=[0.0, 1.0], shape=[10], rng=rng)
    assert not np.array_equal(result1, result3)

    # Default RNG
    ndd.random.set_seed(9876)
    values1 = [ndd.random.get_default_rng()() for _ in range(5)]
    ndd.random.set_seed(9876)
    values2 = [ndd.random.get_default_rng()() for _ in range(5)]
    assert values1 == values2
    ndd.random.set_seed(5432)  # Different seed should produce different values
    values3 = [ndd.random.get_default_rng()() for _ in range(5)]
    assert values1 != values3

    # Default RNG with operators
    ndd.random.set_seed(1111)
    result1 = ndd.random.uniform(range=[0.0, 1.0], shape=[10])
    ndd.random.set_seed(1111)
    result2 = ndd.random.uniform(range=[0.0, 1.0], shape=[10])
    assert np.array_equal(result1, result2)
    ndd.random.set_seed(2222)  # Different seed
    result3 = ndd.random.uniform(range=[0.0, 1.0], shape=[10])
    assert not np.array_equal(result1, result3)


@params(
    (0, (0,)),
    (1, (1, 2)),
    (2, (np.int64(2), np.uint64(3))),
)
def test_rng_advance_matches(offset: int, advances: tuple[SupportsIndex, ...]):
    rng = ndd.random.RNG(seed=7)
    for _ in range(offset):
        rng()
    reference = rng.clone()

    for n in advances:
        rng.advance(n)

    for _ in range(sum(int(n) for n in advances)):
        reference()

    assert [rng() for _ in range(5)] == [reference() for _ in range(5)]
