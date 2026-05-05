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

"""Random number generator support for DALI dynamic mode.

This module also contains functional wrappers for random operators (e.g., uniform, normal)
that are dynamically added during module initialization.
"""

import random as _random
import threading as _threading

import nvidia.dali.backend_impl as _b

# Note: __all__ is intentionally not defined here to allow help() to show
# dynamically added random operator functions (uniform, normal, etc.)


class RNG:
    """Random number generator for DALI dynamic mode operations.

    This RNG can be used to provide reproducible random state to DALI operators.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generator. If not provided, a random seed is used.
        The seed is truncated to 64 bits.
        Mutually exclusive with `state`.
    state : State object, optional
        The state object as returned RNG.state property.
        Mutually exclusive with `seed`.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>>
    >>> # Create an RNG with a specific seed
    >>> my_rng = ndd.random.RNG(seed=1234)
    >>>
    >>> # Use it with random operators
    >>> result = ndd.random.uniform(range=(-1, 1), shape=[10], rng=my_rng, device="cpu")
    """

    def __init__(self, seed=None, state=None):
        if seed is not None:
            if state is not None:
                raise ValueError("Cannot specify both `seed` and `state`")
            self._rng = _b._Philox4x32_10(seed & 0xFFFFFFFFFFFFFFFF, 0, 0)
        elif state is not None:
            self._rng = _b._Philox4x32_10(state)
        else:
            seed = _random.randint(0, 0xFFFFFFFFFFFFFFFF)
            self._rng = _b._Philox4x32_10(seed, 0, 0)

    def __call__(self):
        """Generate a random uint32 value.

        Returns
        -------
        int
            A random uint32 value (as Python int, but in range [0, 2^32-1]).
        """
        return self._rng.next()

    @property
    def seed(self):
        raise AttributeError("seed is write-only")

    @seed.setter
    def seed(self, value):
        """Set the seed for this RNG and reset its random sequence.
        The seed is truncated to 64 bits.
        """
        self._rng = _b._Philox4x32_10(value & 0xFFFFFFFFFFFFFFFF, 0, 0)

    def clone(self):
        """Create a new RNG with the same state

        Returns
        -------
        RNG
            A new RNG which will continue the same sequence as this one.

        Examples
        --------
        >>> import nvidia.dali.experimental.dynamic as ndd
        >>>
        >>> # Create an RNG
        >>> rng1 = ndd.random.RNG(seed=1234)
        >>> _ = rng1()  # change the state of the generator
        >>>
        >>> # Clone it to create an independent copy
        >>> rng2 = rng1.clone()
        >>>
        >>> # Both will generate the same sequence
        >>> for i in range(10):
        >>>     assert rng1() == rng2()
        """
        return RNG(state=self.state)

    def __repr__(self):
        return f"RNG(state='{self.state}')"

    @property
    def state(self):
        """Returns the internal state of the generator.

        Returns
        -------
        Opaque state object. This object can be converted to a string and that string can be used
        later to set the state or construct an RNG.
        """
        return self._rng.get_state()

    @state.setter
    def state(self, value):
        """Sets the internal state of the generator.

        Parameters
        ----------
        value : object | str
            Either a state object obtained from another RNG instance of its string representation
        """
        self._rng.set_state(value)


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
