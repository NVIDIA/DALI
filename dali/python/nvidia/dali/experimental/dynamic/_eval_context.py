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
import sys
import threading
import weakref

from . import _device, _stream
from ._async import _AsyncExecutor
from ._thread_pool import ThreadPool, _get_default_thread_pool


class _ThreadLocalStorage(threading.local):
    def __init__(self):
        super().__init__()
        self.default = {}  # per-device default context
        self.stack = []


_tls = _ThreadLocalStorage()


class EvalContext:
    """
    Evaluation context for DALI dynamic API.

    This class aggregates state and auxiliary objects that are necessary to execute DALI operators.
    These include:

    - CUDA device
    - thread pool
    - cuda stream.

    ``EvalContext`` is a context manager.
    """

    _default_context_stream_sentinel = object()

    def __init__(self, *, num_threads=None, device_id=None, cuda_stream=None, thread_pool=None):
        """
        Constructs an ``EvalContext`` object.

        Keyword Args
        ------------
        thread_pool : ThreadPool, optional
            The thread pool which will be used by multi-threaded operators.
            It must be associated with the same `device_id` as the one passed to this function
            This parameter is mutually exclusive with `num_threads`.
        num_threads : int, optional
            If specified, a new thread pool with this number of threads is created and associated
            with the context. Note that creating a thread pool constitutes considerable overhead.
            This argument is mutually exclusive with `thread_pool`.
        device_id : int, optional
            The ordinal of the GPU associated with the context. If not specified, the current CUDA
            device will be used.
        cuda_stream : stream object, optional
            The cuda_stream on which GPU operators will be executed. If not provided, the value is
            assigned by trying several options, in this order:
            - the thread's default stream, set by calling :meth:`set_current_stream`
            - the default stream set by calling :meth:`set_default_stream`
            - a new stream, if neither of the above was set.
            Compatible streams include:
            - DALI :class:`Stream` class
            - any object exposing ``__cuda_stream__`` interface
            - raw CUDA stream handle
            - PyTorch stream
            see :class:`Stream` for details.
        """
        self._invocations = []
        self._default_stream = None

        if device_id is not None:
            self._device = _device.Device("gpu", device_id)
        else:
            self._device = _device.Device.current()

        if thread_pool is not None:
            if num_threads is not None:
                raise ValueError("`thread_pool` and  `num_threads` cannot be specified together.")
            if thread_pool.device_id != self._device.device_id:
                if device_id is None:
                    device_id_message = f"<Current> ({self._device.device_id})"
                else:
                    device_id_message = device_id
                raise ValueError(
                    f"Device ID clash: device_id == {device_id_message} "
                    f"but thread_pool.device_id == {thread_pool.device_id}"
                )
        elif num_threads is not None:
            thread_pool = ThreadPool(num_threads, device_id=self._device.device_id)

        if cuda_stream is EvalContext._default_context_stream_sentinel:
            self._cuda_stream = None
        else:
            if cuda_stream is not None:
                self._cuda_stream = _stream.stream(stream=cuda_stream, device_id=device_id)
            elif device_id is not None:
                with self._device:
                    self._cuda_stream = _stream.get_current_stream()
            else:  # we're using current device anyway
                self._cuda_stream = _stream.get_current_stream()

        self._instance_cache = {}
        self._num_active = 0
        self._thread_pool = thread_pool

        self._async_executor = _AsyncExecutor()
        weakref.finalize(self, self._async_executor.shutdown)

        # Used to disallow the EvalContext to be active in two threads simultaneously
        self._lock = threading.RLock()

    def _purge_operator_cache(self):
        """Empties the operator instance cache"""
        self._instance_cache = {}

    @property
    def thread_pool(self):
        return self._thread_pool or _get_default_thread_pool(self.device_id)

    @staticmethod
    def current() -> "EvalContext":
        """
        Returns the currently active EvalContext for the calling thread.
        """
        if _tls.stack:
            return _tls.stack[-1]
        else:
            return EvalContext.default()

    @property
    def _is_current(self) -> bool:
        if _tls.stack:
            return self is _tls.stack[-1]

        current_device_id = _device.Device.default_device_id(_device.Device._default_device_type())
        return self is _tls.default.get(current_device_id)

    def __enter__(self):
        skip_lock = self._is_in_background_thread()
        if not skip_lock and not self._lock.acquire(blocking=False):
            raise RuntimeError("An EvalContext cannot be active in two threads simultaneously.")
        self._num_active += 1
        try:
            _tls.stack.append(self)
            if self._device:
                self._device.__enter__()
        except Exception:
            if not skip_lock:
                self._lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._num_active -= 1
        try:
            # During interpreter shutdown, finalizers of objects created in background threads
            # can be called from the main thread.
            if _tls.stack:
                assert _tls.stack[-1] is self
                if self._num_active == 0:
                    self.evaluate_all()
                _tls.stack.pop()
            else:
                assert sys.is_finalizing()
            if self._device:
                self._device.__exit__(exc_type, exc_value, traceback)
        finally:
            if not self._is_in_background_thread():
                self._lock.release()

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
        CUDA device ordinal of the device associated with this ``EvalContext``.
        """
        return self._device.device_id

    @property
    def num_threads(self):
        """
        The number of thread pool workers in this ``EvalContext``.
        """
        return self.thread_pool.num_threads

    @property
    def cuda_stream(self):
        """
        CUDA stream for this ``EvalContext``

        .. note::
            In case of the thread's default context, this value is affected by calls to methods
            :meth:`set_default_stream` and :meth:`set_current_stream`.
        """
        if self._cuda_stream is None:
            s = _stream.get_default_stream(self.device_id)
            if s is None and self._device.device_type == "gpu":
                if self._default_stream is None:
                    self._default_stream = _stream.stream(device_id=self._device.device_id)
                s = self._default_stream
            else:
                self._default_stream = None
            return s
        else:
            return self._cuda_stream

    @staticmethod
    def default() -> "EvalContext":
        """
        The default ``EvalContext`` for the calling thread.
        """
        dev_type = _device.Device._default_device_type()
        current_device_id = _device.Device.default_device_id(dev_type)
        if current_device_id not in _tls.default:
            _tls.default[current_device_id] = EvalContext(
                device_id=current_device_id if dev_type == "gpu" else None,
                cuda_stream=EvalContext._default_context_stream_sentinel,
            )
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
        if ctx._thread_pool is None:
            ctx._thread_pool = self.thread_pool
        return ctx

    def _is_in_background_thread(self):
        return threading.current_thread() is self._async_executor._thread


__all__ = [
    "EvalContext",
]
