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

"""Random number generator support for DALI dynamic mode.

This module also contains functional wrappers for random operators (e.g., uniform, normal)
that are dynamically added during module initialization.
"""

import random as _random
import threading as _threading

# Note: __all__ is intentionally not defined here to allow help() to show
# dynamically added random operator functions (uniform, normal, etc.)


class RNG:
    """Random number generator for DALI dynamic mode operations.

    This RNG can be used to provide reproducible random state to DALI operators.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generator. If not provided, a random seed is used.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>>
    >>> # Create an RNG with a specific seed
    >>> my_rng = ndd.random.RNG(seed=1234)
    >>>
    >>> # Use it with random operators
    >>> result = ndd.ops.random.Uniform(device="cpu")(range=(-1, 1), shape=[10], rng=my_rng)
    """

    def __init__(self, seed=None):
        """Initialize the RNG with an optional seed."""
        if seed is None:
            # Use Python's random to generate a random seed
            seed = _random.randint(0, 2**31 - 1)
        self._rng = _random.Random(seed)
        self._seed = seed

    def __call__(self):
        """Generate a random uint32 value.

        Returns
        -------
        int
            A random uint32 value (as Python int, but in range [0, 2^32-1]).
        """
        # Generate a random value in the uint32 range [0, 2^32-1]
        # Return as Python int (numpy will convert it when creating the array)
        return self._rng.randint(0, 0xFFFFFFFF)

    @property
    def seed(self):
        """Get the seed used to initialize this RNG."""
        return self._seed

    @seed.setter
    def seed(self, value):
        """Set the seed for this RNG and reset its random sequence."""
        self._seed = value
        self._rng = _random.Random(value)

    def clone(self):
        """Create a new RNG with the same seed.

        Returns
        -------
        RNG
            A new RNG instance initialized with the same seed as this one.
            This allows creating independent RNG streams that produce the same
            sequence of random numbers.

        Examples
        --------
        >>> import nvidia.dali.experimental.dynamic as ndd
        >>>
        >>> # Create an RNG
        >>> rng1 = ndd.random.RNG(seed=1234)
        >>>
        >>> # Clone it to create an independent copy
        >>> rng2 = rng1.clone()
        >>>
        >>> # Both will generate the same sequence
        >>> for i in range(10):
        >>>     assert rng1() == rng2()
        """
        return RNG(seed=self._seed)

    def __repr__(self):
        return f"RNG(seed={self._seed})"


# Thread-local storage for the default RNG
_thread_local = _threading.local()


def get_default_rng():
    """Get the default RNG for the current thread.

    Returns
    -------
    RNG
        The default RNG for the current thread.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>>
    >>> # Get the default RNG
    >>> default = ndd.random.get_default_rng()
    >>> print(default)
    """
    if not hasattr(_thread_local, "default_rng"):
        _thread_local.default_rng = RNG()
    return _thread_local.default_rng


def set_seed(seed):
    """Set the seed for the default thread-local RNG.

    This affects all subsequent calls to random operators that don't specify
    an explicit RNG.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>>
    >>> # Set the seed for reproducible results
    >>> ndd.random.set_seed(1234)
    >>> result1 = ndd.random.uniform(range=(-1, 1), shape=[10])
    >>>
    >>> # Reset to the same seed
    >>> ndd.random.set_seed(1234)
    >>> result2 = ndd.random.uniform(range=(-1, 1), shape=[10])
    >>> # result1 and result2 should be identical
    """
    get_default_rng().seed = seed
