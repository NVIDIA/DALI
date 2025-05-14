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

from contextlib import contextmanager
from threading import local

_tls = local()
_tls.stack = []


class EvalContext:

    def __init__(self):
        self._invocations = {}
        self._cached_results = {}
        self.cuda_stream = None

    @staticmethod
    def current():
        return _tls.stack[-1] if _tls.stack else None

    def __enter__(self):
        _tls.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _tls.stack.pop()

    @staticmethod
    def get():
        return EvalContext.current() or EvalContext()

    def cached_results(self, invocation):
        if invocation in self._cached_results:
            return self._cached_results[invocation]

        # TODO(michalz): Common subexpression elimination.
        return None

    def cache_results(self, invocation, results):
        self._cached_results[invocation] = results
