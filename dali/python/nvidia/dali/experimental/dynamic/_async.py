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

import queue
import sys
import threading
from collections.abc import Callable
from typing import Any, Optional


class _Future:
    """Lightweight future created by _AsyncExecutor to track tasks"""

    __slots__ = ("_seq_id", "_executor", "_callable", "_exception")

    def __init__(self, seq_id: int, executor: "_AsyncExecutor", callable: Callable[[], Any]):
        self._seq_id = seq_id
        self._executor = executor
        self._callable: Optional[Callable[[], Any]] = callable
        self._exception: Optional[BaseException] = None

    def _run(self):
        """Call the callback and set the result. Must be called by the executor"""

        if self._callable is None:
            return

        try:
            self._callable()
        except BaseException as exception:
            self._exception = exception
        finally:
            # We need this to break a reference cycle when we are done
            # We don't want to delay releasing GPU memory
            self._callable = None

    def wait(self):
        self._executor.wait(self)

        if self._exception is not None:
            raise self._exception


class _AsyncExecutor:
    """
    Schedule invocation of operators in a background thread.
    It has less overhead than ThreadPoolExecutor because of stronger guarantees (SCSP):
      - Users shouldn't share eval contexts between threads
      - The executor only starts one thread
    """

    def __init__(self):
        self._submitted_seq = -1
        self._completed_seq = -1

        self._condition = threading.Condition()
        self._queue = queue.SimpleQueue[Optional[_Future]]()
        self._thread: Optional[threading.Thread] = None
        self._event = threading.Event()

    def _worker(self):
        while True:
            task = self._queue.get()

            if task is None:
                break

            self._event.set()
            task._run()
            self._event.clear()

            with self._condition:
                self._completed_seq += 1
                self._condition.notify()

    def submit(self, callable: Callable[[], None]):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

        # Since the executor is bound to a single eval context, there's no race condition
        self._submitted_seq += 1
        task = _Future(self._submitted_seq, self, callable)
        self._queue.put(task)

        # Let the worker acquire the GIL if it's ready to pick up a task
        if sys.version_info < (3, 13) or sys._is_gil_enabled():
            self._event.wait(1e-4)  # 100us

        return task

    def wait(self, task: _Future):
        with self._condition:
            self._condition.wait_for(lambda: self._completed_seq >= task._seq_id)

    def shutdown(self):
        self._queue.put(None)
