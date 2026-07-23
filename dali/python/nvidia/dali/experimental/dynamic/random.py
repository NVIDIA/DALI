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

import operator as _operator
import random as _random
import threading as _threading
import typing as _typing

import nvidia.dali.backend_impl as _b

from ._tensor import Tensor as _Tensor
from ._type import uint32 as _uint32

# Note: __all__ is intentionally not defined here to allow help() to show
# dynamically added random operator functions (uniform, normal, etc.)


@_typing.final
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
        self._pending_draws = 0
        self._version = 0
        if state is not None:
            if seed is not None:
                raise ValueError("Cannot specify both `seed` and `state`")
            self.set_state(state)
        else:
            if seed is None:
                seed = _random.randint(0, 0xFFFFFFFFFFFFFFFF)
            self.seed = seed

    def __call__(self):
        """Generate a random uint32 value.

        Returns
        -------
        int
            A random uint32 value (as Python int, but in range [0, 2^32-1]).
        """
        self._flush_pending()
        self._version += 1
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
        self._pending_draws = 0
        self._version += 1

    def advance(self, n: _typing.SupportsIndex) -> None:
        """Advance the generator by ``n`` draws, equivalent to ``n`` calls to ``self()``.

        The advance is deferred and applied lazily on the next observation of the generator
        (a draw, a state read, a clone or a repr), coalescing repeated advances into a single
        backend skip-ahead.

        Parameters
        ----------
        n : int
            A uint64 number of draws to skip.
        """
        n = _operator.index(n)
        if n < 0:
            raise ValueError("Cannot advance by a negative number")

        if n:
            self._pending_draws += n
            self._version += 1

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
        return self.get_state()

    @state.setter
    def state(self, value):
        """Sets the internal state of the generator.

        Parameters
        ----------
        value : object | str
            A state object obtained from another RNG instance or its string representation.
        """
        self.set_state(value)

    def get_state(self, *, cuda_stream=None):
        """Returns the internal state of the generator.

        Equivalent to :attr:`state`. Provided so that an :class:`RNG` exposes the same
        ``get_state`` / ``set_state`` interface as a :class:`Reader`, which is the
        contract used by the checkpointing API.

        Parameters
        ----------
        cuda_stream : Any
            Not used.

        Returns
        -------
        Opaque state object. The object can be converted to a string with ``str(state)`` and
        later used to set the state or construct an RNG.
        """
        self._flush_pending()
        return self._rng.get_state()

    def set_state(self, value):
        """Sets the internal state of the generator.

        Equivalent to assigning to :attr:`state`. Provided so that an :class:`RNG` exposes
        the same ``get_state`` / ``set_state`` interface as a :class:`Reader`, which is
        the contract used by the checkpointing API.

        Parameters
        ----------
        value : object | str
            Either a state object obtained from :func:`get_state` (or the :attr:`state`
            property) or its string representation.
        """
        self._rng = _b._Philox4x32_10(value)
        self._pending_draws = 0
        self._version += 1

    def _flush_pending(self) -> None:
        if pending := self._pending_draws:
            self._rng.skipahead(pending)
            self._pending_draws = 0

    def _snapshot_backend(self):
        return _b._Philox4x32_10(self.state), self._version


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


_STATE_WORDS = 7  # whole uint32 words needed for the backend's 25-byte state


def _draw_state(next_uint32: _typing.Callable[[], int]) -> list[int]:
    return [next_uint32() for _ in range(_STATE_WORDS)]


def _state_tensor(words: list[int]):
    """CPU uint32 tensor holding one drawn random state."""
    return _Tensor(words, dtype=_uint32, device="cpu")


def _resolve_rng(rng: _typing.Any) -> RNG:
    """Return a validated RNG, using the thread-local default when omitted."""
    if rng is None:
        rng = get_default_rng()
    if not isinstance(rng, RNG):
        rng_class = f"{RNG.__module__}.{RNG.__qualname__}"
        user_class = f"{type(rng).__module__}.{type(rng).__qualname__}"
        raise ValueError(f"rng must be an instance of {rng_class}, but got {user_class}")
    return rng
