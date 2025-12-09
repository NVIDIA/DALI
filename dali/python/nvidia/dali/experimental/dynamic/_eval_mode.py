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


import threading
from enum import Enum, auto


class EvalMode(Enum):
    """Enum defining different evaluation modes for Dynamic Mode operations.

    Attributes:
        default:    Default evaluation mode. TBD.
        deferred:   Deferred evaluation mode - operations are evaluated only when their results are
                    needed; error reporting (including input validation) may be delayed until the
                    results are requested.
                    In this mode operations with unused results may be skipped and repeated
                    operations may be merged into one.
        eager:      The evaluation starts immediately. Input validation is immediate.
                    The operations may finish asynchronously.
        sync_cpu:   Synchronous evaluation mode - evaluation on the CPU finishes before the
                    operation returns.
        sync_full:  Fully synchronous evaluation mode - evaluation on all devices finishes before
                    the operation returns.
    """

    deferred = auto()
    eager = auto()
    sync_cpu = auto()
    sync_full = auto()

    default = deferred

    def __enter__(self):
        _tls.eval_mode_stack.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _tls.eval_mode_stack.pop()

    @staticmethod
    def current() -> "EvalMode":
        return _tls.eval_mode_stack[-1]


class _ThreadLocalState(threading.local):
    def __init__(self):
        super().__init__()
        self.eval_mode_stack = [EvalMode.default]


_tls = _ThreadLocalState()

__all__ = ["EvalMode"]
