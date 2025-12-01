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

from threading import local
from . import _device
import nvidia.dali.backend_impl as _b
import weakref

_tls = local()
_tls.default = {}  # per-device default context
_tls.stack = []


def _default_num_threads():
    # TODO(michalz): implement some more elaborate logic here
    return 4


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
        if device_id is not None:
            self._device = _device.Device("gpu", device_id)
        else:
            self._device = _device.Device.current()

        if self._cuda_stream is None and self._device.device_type == "gpu":
            self._cuda_stream = _b.Stream(self._device.device_id)

        self._thread_pool = _b._ThreadPool(num_threads or _default_num_threads())

    @staticmethod
    def current():
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
    def cuda_stream(self):
        """
        CUDA stream for this EvalContext
        """
        return self._cuda_stream

    @staticmethod
    def default():
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


__all__ = ["EvalContext"]
