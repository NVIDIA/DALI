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

import ctypes
import functools
import itertools
from collections.abc import Callable
from typing import Any

import nvidia.dali.experimental.dynamic as ndd
from nose_utils import SkipTest


def eval_modes(*modes: ndd.EvalMode):
    """Automatically run the test function with multiple eval modes."""
    if not modes:
        modes = tuple(ndd.EvalMode)

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


def cuda_launch_host_func(stream: ndd.Stream, func: Callable[[Any], None]):
    """
    Launch a host function on a CUDA stream via ``cudaLaunchHostFunc``.
    Add an attribute to the func to store the callback to prevent it from being garbage collected.
    """
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        raise SkipTest("Could not find libcudart.so") from None

    callback_type = ctypes.PYFUNCTYPE(None, ctypes.c_void_p)
    cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int
    callback = callback_type(func)
    err = cudart.cudaLaunchHostFunc(stream.handle, callback, None)
    assert err == 0, f"cudaLaunchHostFunc failed with error code {err}"

    func._callback = callback
