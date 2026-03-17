# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import threading

import nvidia.dali.backend_impl as _b
from . import _device


class ThreadPool(_b._NewThreadPool):
    CurrentDeviceId = -1

    def __init__(self, num_threads=None, *, device_id=CurrentDeviceId):
        """
        Args
        ----
        num_threads : int, optional
            The number of threads. If not provided, the value returned by
            :meth:`get_num_threads` is used.

        Keyword Args
        ------------
        device_id : int, optional
            The GPU device ordinal to associate with the thread pool. If not provided,
            the current device is used; if None, then no GPU is associated with the thread pool.
        """
        if device_id == ThreadPool.CurrentDeviceId:
            device_id = _device.Device.current().device_id

        if num_threads is None:
            num_threads = get_num_threads()

        super().__init__(num_threads, device_id=device_id)
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id


def _default_num_threads():
    """Gets the default number of threads used in DALI dynamic mode."""
    import os
    from functools import wraps

    mod = sys.modules[__name__]

    if nenv := os.environ.get("DALI_NUM_THREADS", None):
        n = int(nenv)
    else:
        n = len(os.sched_getaffinity(0))

    @wraps(_default_num_threads)
    def __default_num_threads():
        return n

    # The default number of threads will not change, so we replace this costly function
    # with a simple one that returns the constant value we've just determined.
    mod._default_num_threads = __default_num_threads
    return n


_global_num_threads = None
_global_default_thread_pool = {}


class _DefaultThreadPool:
    def __init__(self, device_id):
        self._mutex = threading.Lock()
        self._device_id = device_id
        self._thread_pool = None

    def get(self, num_threads):
        tp = self._thread_pool
        if tp is None or tp.num_threads != num_threads:
            with self._mutex:
                if self._thread_pool is None or self._thread_pool.num_threads != num_threads:
                    tp = self._thread_pool = ThreadPool(num_threads, device_id=self._device_id)
        return tp

    def _set_num_threads(self, num_threads):
        tp = self._thread_pool
        # This check can be made outside of the lock - calling _set_num_threads from multiple
        # threads is UB anyway
        if tp is not None and tp.num_threads != num_threads:
            with self._mutex:  # this prevents `get` from returning None
                self._thread_pool = None


def _init_thread_pool_dict():
    _global_default_thread_pool[None] = _DefaultThreadPool(None)
    for i in range(_b.GetCUDADeviceCount()):
        _global_default_thread_pool[i] = _DefaultThreadPool(i)


_init_thread_pool_dict()


def _get_default_thread_pool(device_id):
    """Returns the global default thread pool, creating or recreating it if needed."""
    return _global_default_thread_pool[device_id].get(get_num_threads())


def get_num_threads():
    """
    Gets the number of threads in the default thread pool.

    The value is determined by (in decreasing priority):
    1. The value (not None) passed to :meth:`set_num_threads`
    2. The value from DALI_NUM_THREADS environment variable.
    3. The number of CPUs in the calling process affinity list: ``len(os.sched_getaffinity(0))``
    """
    return _global_num_threads or _default_num_threads()


def set_num_threads(n):
    """
    Sets (or clears) the number of threads in the default thread pool.

    Changing this value will cause all EvalContexts which were constructed without an explicitly
    given number of threads to recreate their associated thread pools.

    Setting None will cause the default value to be used.

    The value must be a positive integer and must not exceed 100 threads per CPU.

    .. warning::
        This function should be called once, at the beginning of the program.
        Changing this value later is very costly and should be avoided.

    .. warning::
        This function is not thread safe - calling `set_num_threads` concurrently
        constitutes a race condition and results in an undefined behavior.
    """
    global _global_num_threads

    if n is None:
        _global_num_threads = None
        new_count = get_num_threads()
    elif not isinstance(n, int):
        raise TypeError("The number of threads must be an integer")
    elif n <= 0:
        raise ValueError(f"The number of threads must be positive; got {n}.")
    else:
        new_count = n

    import multiprocessing

    if new_count > multiprocessing.cpu_count() * 100:
        raise ValueError(
            f"The number of threads per CPU core must not exceed 100.\n"
            f"Got {new_count} threads for {multiprocessing.cpu_count()} cores."
        )

    for tp in _global_default_thread_pool.values():
        tp._set_num_threads(new_count)

    if n is not None:  # otherwise keep cleared
        _global_num_threads = new_count


__all__ = [
    "ThreadPool",
    "set_num_threads",
    "get_num_threads",
]
