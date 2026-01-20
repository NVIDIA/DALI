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

import copy
import weakref
from threading import local

import nvidia.dali.backend_impl as _b

from . import _device
from ._async import _AsyncExecutor


_global_streams = None


class _ThreadLocalStorage(local):
    def __init__(self):
        super().__init__()
        self.default = {}  # per-device default context
        self.stack = []


_tls = _ThreadLocalStorage()


def _default_num_threads():
    """Gets the default number of threads used in DALI dynamic mode."""
    import sys
    import os
    from functools import wraps

    mod = sys.modules[__name__]

    if nenv := os.environ.get("DALI_NUM_THREADS", None):
        n = int(nenv)
    else:
        n = len(os.sched_getaffinity(0))

    print("NDD: default number of threads", n)

    @wraps(_default_num_threads)
    def __default_num_threads():
        return n

    mod._default_num_threads = __default_num_threads
    return n


_global_num_threads = None


def _get_stream_device(s):
    if s is None:
        return None
    dev_id = getattr(s, "device_id", None)
    if dev_id is not None:
        return dev_id
    dev = getattr(s, "device", None)
    if dev is not None:
        if isinstance(dev, int):
            return dev
        idx = getattr(dev, "index", None)
        if idx is not None:
            return idx
        id = getattr(dev, "id", None)
        if id is not None:
            return id
    return _b.GetCUDAStreamDevice(s)


def set_stream(stream, device_id=None):
    """
    Sets the default stream for a CUDA device. If the device id is not specified,
    the device associated with the stream will be used. If there's no device associated with the
    stream, current CUDA device is used.
    """
    global _global_streams
    if not _global_streams:
        _global_streams = [None] * _b.GetCUDADeviceCount()
    if stream is not None:
        stream_dev = _get_stream_device(stream)
        if device_id is not None and stream_dev is not None and device_id != stream_dev:
            raise ValueError(
                f"The device_id {device_id} doesn't match the device associated "
                f"with the stream {stream_dev}"
            )
        if device_id is None:
            device_id = stream_dev

    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()

    _global_streams[device_id] = stream


def get_stream(device_id=None):
    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()
    return _global_streams[device_id]


def get_num_threads():
    """
    Gets the number of threads in the default thread pool.

    The value is determined by (in decreasing priority):
    1. The value (not None) passed to :meth:`set_num_threads``
    2. The value from DALI_NUM_THREADS environment variable.
    3. The number of CPUs in the calling process affinity list: ``len(os.sched_getaffinity(0))``
    """
    return _global_num_threads or _default_num_threads()


def set_num_threads(n):
    """
    Sets (or clears) the number of threads in the default thread pool.

    Changing this value will cause all EvalCotnexts which were constructed without an explicitly
    given number of threads to recreate their associated thread pools.

    Setting None will cause the default value to be used.

    The value must be a positive integer and must not exceed 100 threads per CPU.
    """
    global _global_num_threads
    if n is None:
        _global_num_threads = None
        return
    if not isinstance(n, int):
        raise TypeError("The number of threads must be an integer")
    if n <= 0:
        raise ValueError(f"The number of threads must be positive; got {n}.")
    import multiprocessing

    if n > multiprocessing.cpu_count() * 100:
        raise ValueError(
            f"The number of threads per CPU core must not exceed 100.\n"
            f"Got {n} threads for {multiprocessing.cpu_count()} cores."
        )

    _global_num_threads = n


class EvalContext:
    """
    Evaluation context for DALI dynamic API.

    This class aggregates state and auxiliary objects that are necessary to execute DALI operators.
    These include:

    - CUDA device
    - thread pool
    - cuda stream.

    EvalContext is a context manager.
    """

    def __init__(self, *, num_threads=None, device_id=None, cuda_stream=None):
        """
        Constructs an EvalContext object.

        Keyword Args
        ------------
        num_threads : int, optional
            The number of threads in the new thread pool that will be associated with the context.
        device_id : int, optional
            The ordinal of the GPU associated with the context. If not specified, the current CUDA
            device will be used.
        cuda_stream : dali.backend.Stream, __cuda_stream__ interface or raw stream handle, optional
            The cuda_stream on which GPU operators will be executed. If not provided, a new stream
            will be created.
        """
        self._invocations = []
        self._cuda_stream = cuda_stream
        self._default_stream = None
        if device_id is not None:
            self._device = _device.Device("gpu", device_id)
        else:
            self._device = _device.Device.current()

        self._num_threads = num_threads
        self._instance_cache = {}

        # The thread pool needs to be thread-local because of eager execution
        self._tls = local()

        self._async_executor = _AsyncExecutor()
        weakref.finalize(self, self._async_executor.shutdown)

    @property
    def _thread_pool(self):
        if (
            not hasattr(self._tls, "thread_pool")
            or self._tls.thread_pool.num_threads != self.num_threads
        ):
            self._tls.thread_pool = _b._ThreadPool(self.num_threads)

        return self._tls.thread_pool

    @staticmethod
    def current() -> "EvalContext":
        """
        Returns the currently active EvalContext for the calling thread.
        """
        if _tls.stack:
            return _tls.stack[-1]
        else:
            return EvalContext.default()

    def __enter__(self):
        _tls.stack.append(self)
        if self._device:
            self._device.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert _tls.stack[-1] is self
        if self._device:
            self._device.__exit__(exc_type, exc_value, traceback)
        _tls.stack.pop()
        if EvalContext.current() is not self:
            self.evaluate_all()

    def evaluate_all(self):
        """Evaluates all pending invocations."""
        tmp = self._invocations
        self._invocations = []  # prevent recursive invocation
        for weak_inv in tmp:
            inv = weak_inv() if isinstance(weak_inv, weakref.ReferenceType) else weak_inv
            if inv is not None:
                r = self.cached_results(inv)
                if r is not None:
                    continue
                inv.run(self)

    @property
    def device_id(self):
        """
        CUDA device ordinal of the device associated with this EvalContext.
        """
        return self._device.device_id

    @property
    def num_threads(self):
        """
        The number of thread pool workers in this EvalContext.

        If the value was not specified at construction, :meth:`get_num_threads` is used.
        """
        return self._num_threads or get_num_threads()

    @property
    def cuda_stream(self):
        """
        CUDA stream for this EvalContext
        """
        if self._cuda_stream is None:
            s = _global_streams[self.device_id] if _global_streams else None
            if s is None and self._device.device_type == "gpu":
                if self._default_stream is None:
                    self._default_stream = _b.Stream(self._device.device_id)
                s = self._default_stream
            else:
                self._default_stream = None
            return s
        else:
            return self._cuda_stream

    @staticmethod
    def default() -> "EvalContext":
        current_device_id = _device.Device.default_device_id("gpu")
        if current_device_id not in _tls.default:
            _tls.default[current_device_id] = EvalContext(device_id=current_device_id)
        return _tls.default[current_device_id]

    def cached_results(self, invocation):
        """
        Reserved for future use
        """
        # TODO(michalz): Implement something that doesn't leak memory
        # if invocation in self._cached_results:
        #     return self._cached_results[invocation]

        # TODO(michalz): Common subexpression elimination.
        return None

    def cache_results(self, invocation, results):
        """
        Reserved for future use
        """
        pass  # TODO(michalz): Implement something that doesn't leak memory

    def _add_invocation(self, invocation, weak=True):
        self._invocations.append(weakref.ref(invocation) if weak else invocation)

    def _snapshot(self):
        ctx = copy.copy(self)
        if ctx._cuda_stream is None:
            ctx._cuda_stream = self.cuda_stream
        if ctx._num_threads is None:
            ctx._num_threads = self.num_threads
        return ctx


__all__ = [
    "EvalContext",
    "get_stream",
    "set_stream",
    "get_num_threads",
    "set_num_threads",
]
