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

import functools
import itertools

import nvidia.dali.experimental.dynamic as ndd

ALL_EVAL_MODES = (
    ndd.EvalMode.deferred,
    ndd.EvalMode.eager,
    ndd.EvalMode.sync_cpu,
    ndd.EvalMode.sync_full,
)


def eval_modes(*modes: ndd.EvalMode):
    """Automatically run the test function with multiple eval modes."""
    if not modes:
        modes = ALL_EVAL_MODES

    def decorator(func):
        @functools.wraps(func)
        def wrapper(eval_mode, *args, **kwargs):
            with eval_mode:
                return func(*args, **kwargs)

        existing_params = getattr(func, "paramList", [()])
        wrapper.paramList = tuple(
            (mode,) + (p if isinstance(p, tuple) else (p,))
            for mode, p in itertools.product(modes, existing_params)
        )
        return wrapper

    return decorator
