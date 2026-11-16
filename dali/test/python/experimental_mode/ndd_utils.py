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
import ctypes
import functools
import gc
import itertools
from collections.abc import Callable
from typing import Any, TypeGuard

from nose_utils import SkipTest

import nvidia.dali.experimental.dynamic as ndd
from nvidia.dali.experimental.dynamic._capture import CapturedBatch


def _is_captured(batch) -> TypeGuard[CapturedBatch]:
    return isinstance(batch, CapturedBatch)


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


@contextlib.contextmanager
def gc_disabled(stream: ndd.Stream | None = None):
    """
    Keep the cyclic garbage collector disabled while a Python host function may be pending.

    A host function launched with :func:`cuda_launch_host_func` needs the GIL to run and to return,
    so it cannot complete while another thread holds the GIL and waits for the device. A garbage
    collection can do exactly that: it holds the GIL and may finalize objects left behind by
    earlier tests whose destructors synchronize the device (e.g. GPU image decoders freeing their
    memory), and the process deadlocks.

    Disabling the collector is what prevents this. Garbage is collected once on entry, before any
    host function is launched, so that pending finalizers run while that is still safe.
    If ``stream`` is given, it is synchronized on exit (also when the block raises) before the
    collector is re-enabled, so no host function launched on it is left pending.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.collect()
    gc.disable()
    try:
        yield
    finally:
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            if was_enabled:
                gc.enable()


def cuda_launch_host_func(stream: ndd.Stream, func: Callable[[Any], None]):
    """
    Launch a host function on a CUDA stream via ``cudaLaunchHostFunc``.
    Add an attribute to the func to store the callback to prevent it from being garbage collected.

    The host function runs Python code, so it needs the GIL: launch it and wait for it inside
    :func:`gc_disabled` to avoid a deadlock with garbage collection.
    """
    assert not gc.isenabled(), "Launch Python host functions inside gc_disabled()"
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
