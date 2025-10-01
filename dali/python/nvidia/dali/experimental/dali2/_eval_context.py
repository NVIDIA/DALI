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
_tls.default = None
_tls.stack = []

default_num_threads = 4


class EvalContext:

    def __init__(self, num_threads=None, device_id=None, cuda_stream=None):
        self._invocations = []
        self._cached_results = {}
        self._cuda_stream = cuda_stream
        if device_id is None:
            try:
                device_id = _b.GetCUDACurrentDevice()
            except Exception:
                device_id = None
        if device_id is not None:
            self._device = _device.Device("gpu", device_id)
        else:
            self._device = _device.Device.current()

        if self._cuda_stream is None and self._device.device_type == "gpu":
            self._cuda_stream = _b.Stream(self._device.device_id)

        self._thread_pool = _b._ThreadPool(num_threads or default_num_threads)

    @staticmethod
    def current():
        return _tls.stack[-1] if _tls.stack else EvalContext.default()

    def __enter__(self):
        _tls.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
        return self._device.device_id

    @property
    def cuda_stream(self):
        return self._cuda_stream

    @staticmethod
    def get():
        return EvalContext.current() or EvalContext.default()

    @staticmethod
    def default():
        if _tls.default is None:
            _tls.default = EvalContext()
        return _tls.default

    def cached_results(self, invocation):
        if invocation in self._cached_results:
            return self._cached_results[invocation]

        # TODO(michalz): Common subexpression elimination.
        return None

    def cache_results(self, invocation, results):
        self._cached_results[invocation] = results

    def _add_invocation(self, invocation, weak=True):
        self._invocations.append(weakref.ref(invocation) if weak else invocation)


__all__ = ["EvalContext"]
