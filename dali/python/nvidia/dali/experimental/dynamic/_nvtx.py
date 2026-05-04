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
import os
import nvtx

_DOMAIN = nvtx.get_domain("DALI")

_NVTX_ENABLED = bool(
    os.environ.get("NVTX_INJECTION64_PATH")
    or os.environ.get("NSYS_INJECTION_PATH")
    or os.environ.get("CUDA_INJECTION64_PATH")
)

if _NVTX_ENABLED:

    class NVTXRange(contextlib.ContextDecorator):
        """
        NVTX range marker for the DALI domain. Can be used as a context manager or decorator.
        Use categories to organize annotations.
        Note that this class is used only if a profiler is detected; otherwise it's replaced with
        a no-op stub.
        """

        def __init__(self, message: str, color: int | str = 0x957DAD, category: str | None = None):
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

else:

    class NVTXRange(contextlib.ContextDecorator):
        """
        NVTX range marker for the DALI domain. Can be used as a context manager or decorator.
        Use categories to organize annotations. This variant is no-op and is used when profiling
        is disabled.
        """

        def __init__(self, message: str, color: int | str = 0x957DAD, category: str | None = None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
