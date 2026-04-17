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

_transparent_codes: set[types.CodeType] = set()


def mark_transparent(func: types.FunctionType) -> types.FunctionType:
    """Mark a function as transparent plumbing (not a semantic call site).

    Transparent frames are skipped by :func:`resolve_callsite_frame`.
    Follows ``__wrapped__`` decorator chains.
    """
    _transparent_codes.add(func.__code__)
    wrapped = getattr(func, "__wrapped__", None)
    if wrapped is not None:
        mark_transparent(wrapped)
    return func


def resolve_callsite_frame(frame: types.FrameType | None = None) -> types.FrameType | None:
    """Walk ``f_back`` from `frame`, skipping transparent frames.
    The call site is typically at most a few frames above so this shouldn't be too expensive.
    """
    if frame is None:
        frame = sys._getframe(1)
    while frame is not None and frame.f_code in _transparent_codes:
        frame = frame.f_back
    return frame


def capture_stack_from_frame(frame: types.FrameType) -> list[tuple[types.CodeType, int]]:
    """Capture a call stack starting from *frame*"""
    stack: list[tuple[types.CodeType, int]] = []
    current: types.FrameType | None = frame
    while current is not None:
        stack.append((current.f_code, current.f_lineno))
        current = current.f_back
    return stack
