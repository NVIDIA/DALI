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

import contextlib
import enum

import nvtx

_DOMAIN = nvtx.get_domain("DALI")


# Note: sync with dali/core/nvtx.h
class Color(enum.IntEnum):
    RED = 0xFF0000
    GREEN = 0x00FF00
    BLUE = 0x0000FF
    YELLOW = 0xB58900
    ORANGE = 0xCB4B16
    RED1 = 0xDC322F
    MAGENTA = 0xD33682
    VIOLET = 0x6C71C4
    BLUE1 = 0x268BD2
    CYAN = 0x2AA198
    GREEN1 = 0x859900
    NVGREEN = 0x76B900


class NVTXRange(contextlib.ContextDecorator):
    """
    NVTX range marker for the DALI domain. Can be used as a context manager or decorator.
    Use categories to organize annotations.
    """

    def __init__(self, message: str, color: Color = Color.CYAN, category: str | None = None):
        category_id = _DOMAIN.get_category_id(category)
        self._attributes = _DOMAIN.get_event_attributes(
            message=message,
            color=color,
            category=category_id,
        )

    def __enter__(self):
        _DOMAIN.push_range(self._attributes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _DOMAIN.pop_range()
