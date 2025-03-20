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
        self._expressions = {}

    def current(self):
        return self._tls.stack[-1] if self._tls.stack else None

    def __enter__(self):
        _tls.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _tls.stack.pop()

    @staticmethod
    def get():
        return EvalContext.current() or EvalContext()

