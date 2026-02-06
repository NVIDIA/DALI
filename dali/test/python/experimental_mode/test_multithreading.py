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
import os
import sys
import threading
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import cartesian_params, params
from nose_utils import SkipTest


def allow_nogil_failure(exc_type: type[Exception]):
    """
    Skip the test on free-threaded Python if a specific exception is raised.
    This is useful until https://github.com/python/cpython/pull/133305 is backported.
    """

    def decorator(test_func):
        if getattr(sys, "_is_gil_enabled", lambda: True)():
            return test_func

        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            try:
                return test_func(*args, **kwargs)
            except exc_type:
                raise SkipTest(f"{exc_type.__name__} allowed for this test with the GIL disabled")

        return wrapper

    return decorator


T = TypeVar("T")


def run_parallel(function: Callable[[int], T], num_threads: int | None = None) -> dict[int, T]:
    if num_threads is None:
        try:
            num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            num_threads = os.cpu_count() or 4

    barrier = threading.Barrier(num_threads)
    results = {}
    errors = {}

    def wrapper(thread_id: int):
        try:
            barrier.wait()
            results[thread_id] = function(thread_id)
        except Exception as exception:
            errors[thread_id] = exception

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=wrapper, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if errors:
        # Raise only the first error
        thread_id, error = next(iter(errors.items()))
        error.args = (f"Error on thread {thread_id}: {error}", *error.args[1:])
        raise error

    return results


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_parallel_eval_contexts(device):
    def worker(thread_id: int):
        with ndd.EvalContext() as ctx:
            assert ndd.EvalContext.current() is ctx

            data = np.array([thread_id * 10 + i for i in range(5)], dtype=np.float32)
            tensor = ndd.as_tensor(data, device=device)
            result = tensor * 2 + thread_id

            expected = data * 2 + thread_id
            return result.evaluate(), expected

    results = run_parallel(worker)

    for actual, expected in results.values():
        np.testing.assert_equal(actual.cpu(), expected)


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_parallel_creation(device):
    def worker(thread_id: int):
        tensor_data = np.arange(thread_id * 100, thread_id * 100 + 10)
        tensor = ndd.as_tensor(tensor_data, device=device)

        batch_data = [np.array([thread_id, i]) for i in range(3)]
        batch = ndd.as_batch(batch_data, device=device)

        return {
            "tensor": tensor.evaluate(),
            "tensor_expected": tensor_data,
            "batch": batch.evaluate(),
            "batch_expected": batch_data,
        }

    results = run_parallel(worker)

    for data in results.values():
        np.testing.assert_array_equal(data["tensor"].cpu(), data["tensor_expected"])
        assert data["batch"].batch_size == len(data["batch_expected"])
        for actual, expected in zip(data["batch"], data["batch_expected"]):
            np.testing.assert_array_equal(actual.cpu(), expected)


@allow_nogil_failure(KeyError)
def test_parallel_different_devices():
    def worker(thread_id: int):
        device = "cpu" if thread_id % 2 == 0 else "gpu"

        data = np.array([thread_id] * 5, dtype=np.float32)
        tensor = ndd.as_tensor(data, device=device)
        result = tensor + 1
        assert result.device.device_type == device

        expected = data + 1
        return result.evaluate(), expected

    results = run_parallel(worker)

    for result, expected in results.values():
        np.testing.assert_equal(result.cpu(), expected)


@allow_nogil_failure(KeyError)
@cartesian_params(("cpu", "gpu"), ndd.EvalMode)
def test_parallel_eval_modes(device, eval_mode):
    def worker(thread_id: int):
        with eval_mode:
            assert ndd.EvalMode.current() == eval_mode

            tensor = ndd.tensor([thread_id + 1.0] * 4, device=device)
            result = ndd.math.sqrt(tensor)

            expected = np.sqrt(tensor.cpu())
            return result.evaluate(), expected

    results = run_parallel(worker)

    for actual, expected in results.values():
        np.testing.assert_array_almost_equal(actual.cpu(), expected)


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_parallel_mixed_eval_modes(device):
    eval_modes = tuple(ndd.EvalMode)

    def worker(thread_id: int):
        mode = eval_modes[thread_id % len(eval_modes)]
        with mode:
            assert ndd.EvalMode.current() == mode

            data = np.array([thread_id + 1.0] * 3, dtype=np.float32)
            t = ndd.tensor(data, device=device)
            result = t * t

            expected = data * data

            return {
                "mode": mode,
                "result": result,
                "expected": expected,
            }

    results = run_parallel(worker)
    for data in results.values():
        np.testing.assert_array_almost_equal(data["result"].cpu(), data["expected"])


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_parallel_indexing(device):
    tensor = ndd.tensor([[1, 2, 3], [4, 5, 6]], device=device)

    def worker(thread_id: int):
        i = j = thread_id
        i %= tensor.shape[0]
        j %= tensor.shape[1]

        slice = (i, j)
        item = tensor.cpu()[slice].item()

        return slice, item

    results = run_parallel(worker)
    for slice, result in results.values():
        assert result == tensor.cpu()[slice].item()


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_thread_local_rng_determinism(device):
    def worker(_):
        rng = ndd.random.RNG(seed=seed)
        uniform = ndd.random.uniform(range=(0.0, 1.0), shape=[10], rng=rng, device=device)
        normal = ndd.random.normal(mean=0.0, stddev=1.0, shape=[10], rng=rng, device=device)

        return {
            "uniform": uniform.evaluate(),
            "normal": normal.evaluate(),
        }

    seed = 12345
    results = run_parallel(worker).values()
    reference = worker(None)

    for data in results:
        np.testing.assert_array_equal(data["uniform"].cpu(), reference["uniform"].cpu())
        np.testing.assert_array_equal(data["normal"].cpu(), reference["normal"].cpu())


@allow_nogil_failure(KeyError)
@params("cpu", "gpu")
def test_chained_threads(device):
    source = ndd.tensor([1, 2, 3, 4], dtype=ndd.float32, device=device).evaluate()
    result = None

    def worker1(tensor: ndd.Tensor):
        nonlocal result
        result = ndd.math.sqrt(tensor)

    def worker2(tensor: ndd.Tensor):
        nonlocal result
        result = ndd.math.pow(tensor, 2)

    thread1 = threading.Thread(target=worker1, args=(source,))
    thread1.start()
    thread1.join()

    thread2 = threading.Thread(target=worker2, args=(result,))
    thread2.start()
    thread2.join()

    assert result is not None
    np.testing.assert_array_almost_equal(result.cpu(), source.cpu())
