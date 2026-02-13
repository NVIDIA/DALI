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
import types
from collections.abc import Sequence
from typing import NoReturn, TypeAlias, TypeVar

from ._eval_mode import EvalMode


class DisplacedEvaluationError(Exception):
    """Base class for exceptions related to EvalMode.deferred and EvalMode.eager"""

    def __init__(self, base: Exception, message: str):
        super().__init__(message)
        self.__traceback__ = base.__traceback__

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # strip internal names from the qualname
        cls.__module__, *_ = cls.__module__.rsplit("._")


class DeferredEvaluationError(DisplacedEvaluationError):
    def __init__(self, base: Exception):
        super().__init__(base, "An error happened during deferred evaluation")


class AsynchronousExecutionError(DisplacedEvaluationError):
    def __init__(self, base: Exception):
        super().__init__(base, "An error happened during asynchronous execution")


T = TypeVar("T", bound=BaseException)
CallStack: TypeAlias = Sequence[tuple[types.CodeType, int]]

_TEMPLATE_CODE = (lambda: sys._getframe(0)).__code__


def _make_frame(code: types.CodeType, lineno: int) -> types.FrameType:
    # It's currently impossible to create a blank frame in pure Python code.
    # We use _TEMPLATE_CODE instead to get a code object returning a frame and patch it.
    # This relies on documented Python features and should not be implementation-specific
    new_code = _TEMPLATE_CODE.replace(
        co_filename=code.co_filename,
        co_name=code.co_name,
        co_firstlineno=lineno,
    )
    func = types.FunctionType(new_code, {"sys": sys})
    return func()


def _make_traceback(stack: CallStack) -> types.TracebackType | None:
    tb = None
    for code, lineno in stack:
        tb = types.TracebackType(tb, _make_frame(code, lineno), 0, lineno)

    return tb


def rethrow_exception(old_exception: T, stack: CallStack, eval_mode: EvalMode) -> NoReturn:
    """
    Create a new exception with a synthetic traceback and change the type of the initial one.
    Raise the old one with the changed type from the newly created one.
    """
    if eval_mode is EvalMode.eager:
        exception = AsynchronousExecutionError(old_exception)
    else:  # eval_mode is EvalMode.deferred
        exception = DeferredEvaluationError(old_exception)
    traceback = _make_traceback(stack)
    source = old_exception.with_traceback(traceback)

    raise exception from source


def capture_stack(depth: int, limit: int | None = None) -> CallStack:
    """
    Capture a call stack.
    Returns a sequence of code objects and line numbers instead of e.g. traceback.StackSummary
    because of efficiency concerns.
    """
    frame = sys._getframe(depth + 1)
    stack = []
    n = 0
    while frame is not None:
        if limit is not None and n >= limit:
            break
        stack.append((frame.f_code, frame.f_lineno))
        frame = frame.f_back
        n += 1
    return stack
