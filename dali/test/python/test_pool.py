# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._multiproc.pool import WorkerPool
from nvidia.dali._multiproc.messages import TaskArgs, SampleRange
from contextlib import closing
from nvidia.dali._utils.external_source_impl import get_callback_from_source
from nvidia.dali.types import SampleInfo
from functools import wraps
import numpy as np
import os
from nose_utils import raises, with_setup

from test_pool_utils import capture_processes, setup_function, teardown_function


def answer(pid, info):
    return np.array([pid, info.idx_in_epoch, info.idx_in_batch, info.iteration])


def simple_callback(info):
    pid = os.getpid()
    return answer(pid, info)


def another_callback(info):
    return simple_callback(info) + 100


class IteratorCb:
    def __init__(self):
        self.count = 0
        self.pid = None

    def __iter__(self):
        self.pid = os.getpid()
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        return [np.array([self.pid, self.count]) for i in range(self.count)]


class MockGroup:
    def __init__(self, source_desc, batch, prefetch_queue_depth, bytes_per_sample_hint):
        self.source_desc = source_desc
        self.batch = batch
        self.prefetch_queue_depth = prefetch_queue_depth
        self.bytes_per_sample_hint = bytes_per_sample_hint

    @classmethod
    def from_callback(
        cls, callback, batch=False, prefetch_queue_depth=1, bytes_per_sample_hint=None
    ):
        _, source_desc = get_callback_from_source(callback, cycle=None)
        return cls(source_desc, batch, prefetch_queue_depth, bytes_per_sample_hint)


def create_pool(groups, keep_alive_queue_size=1, num_workers=1, start_method="fork"):
    pool = WorkerPool.from_groups(
        groups, keep_alive_queue_size, start_method=start_method, num_workers=num_workers
    )
    try:
        capture_processes(pool)
        return closing(pool)
    except Exception:
        pool.close()
        raise


def get_pids(worker_pool):
    # Note that we also capture the pids so the setup_function and teardown_function can
    # verify its correctness.
    return worker_pool.pids()


def assert_scheduled_num(context, num_tasks):
    assert len(context.partially_received) == num_tasks
    assert len(context.scheduled_minibatches) == num_tasks
    assert len(context.task_queue) == num_tasks


start_methods = ["fork", "spawn"]

# Invoke the `fn` with all start methods. Call setup and teardown before and after the test.
#
# We do this to not repeat the pattern of:
#
# def check_something(start_method):
#    ...
#
# @with_setup(setup_function, teardown_function)
# def test_something():
#   for start_method in start_methods:
#      yield check_something, start_method


def check_pool(fn):
    @wraps(fn)
    def wrapper():
        for start_method in start_methods:
            setup_function()
            yield fn, start_method
            teardown_function()

    return wrapper


# ################################################################################################ #
# 1 callback, 1 worker tests
# ################################################################################################ #


@check_pool
def test_pool_one_task(start_method):
    groups = [MockGroup.from_callback(simple_callback)]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=1, start_method=start_method
    ) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(0, 0, 0, 0),)]
        work_batch = TaskArgs.make_sample(SampleRange(0, 1, 0, 0))
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        batch = pool.receive_batch(context_i=0)
        for task, sample in zip(tasks, batch):
            np.testing.assert_array_equal(answer(pid, *task), sample)


@check_pool
def test_pool_multi_task(start_method):
    groups = [MockGroup.from_callback(simple_callback)]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=1, start_method=start_method
    ) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(i, i, 0, 0),) for i in range(10)]
        work_batch = TaskArgs.make_sample(SampleRange(0, 10, 0, 0))
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        batch = pool.receive_batch(context_i=0)
        for task, sample in zip(tasks, batch):
            np.testing.assert_array_equal(answer(pid, *task), sample)


# Test that we can safely hold as many results as the keep_alive_queue_size
@check_pool
def test_pool_no_overwrite_batch(start_method):
    groups = [MockGroup.from_callback(simple_callback, prefetch_queue_depth=0)]
    for depth in [1, 2, 4, 8]:
        with create_pool(
            groups, keep_alive_queue_size=depth, num_workers=1, start_method=start_method
        ) as pool:
            pids = get_pids(pool)
            pid = pids[0]
            work_batches = [TaskArgs.make_sample(SampleRange(i, i + 1, i, 0)) for i in range(depth)]
            task_list = [[(SampleInfo(i, 0, i, 0),)] for i in range(depth)]
            for i, work_batch in enumerate(work_batches):
                pool.schedule_batch(context_i=0, work_batch=work_batch)
            assert_scheduled_num(pool.contexts[0], depth)
            batches = []
            for i in range(depth):
                batches.append(pool.receive_batch(context_i=0))
                assert_scheduled_num(pool.contexts[0], depth - 1 - i)
            tasks_batches = zip(task_list, batches)
            for tasks, batch in tasks_batches:
                for task, sample in zip(tasks, batch):
                    np.testing.assert_array_equal(answer(pid, *task), sample)


# ################################################################################################ #
# 1 callback, multiple workers tests
# ################################################################################################ #


@check_pool
def test_pool_work_split_multiple_tasks(start_method):
    callbacks = [MockGroup.from_callback(simple_callback)]
    with create_pool(
        callbacks, keep_alive_queue_size=1, num_workers=2, start_method=start_method
    ) as pool:
        num_tasks = 16
        pids = get_pids(pool)
        assert len(pids) == 2
        work_batch = TaskArgs.make_sample(SampleRange(0, num_tasks, 0, 0))
        tasks = [(SampleInfo(i, i, 0, 0),) for i in range(num_tasks)]
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        batch = pool.receive_batch(context_i=0)
        for task, sample in zip(tasks, batch):
            np.testing.assert_array_equal(answer(-1, *task)[1:], sample[1:])


# ################################################################################################ #
# multiple callbacks
# ################################################################################################ #


@check_pool
def test_pool_iterator_dedicated_worker(start_method):
    groups = [
        MockGroup.from_callback(simple_callback, prefetch_queue_depth=3),
        MockGroup.from_callback(IteratorCb(), prefetch_queue_depth=3, batch=True),
    ]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=4, start_method=start_method
    ) as pool:
        pids = get_pids(pool)
        assert len(pids) == 4
        tasks_list = []
        samples_count = 0
        for i in range(4):
            tasks = [(SampleInfo(samples_count + j, j, i, 0),) for j in range(i + 1)]
            tasks_list.append(tasks)
            work_batch = TaskArgs.make_sample(
                SampleRange(samples_count, samples_count + i + 1, i, 0)
            )
            samples_count += len(tasks)
            pool.schedule_batch(context_i=0, work_batch=work_batch)
            pool.schedule_batch(context_i=1, work_batch=TaskArgs.make_batch((i,)))
        assert pool.contexts[0].dedicated_worker_id is None
        iter_worker_num = pool.contexts[1].dedicated_worker_id
        iter_worker_pid = pool.pool._processes[iter_worker_num].pid
        for i in range(4):
            batch_0 = pool.receive_batch(context_i=0)
            batch_1 = pool.receive_batch(context_i=1)
            tasks = tasks_list[i]
            assert len(batch_0) == len(tasks)
            assert len(batch_1) == len(tasks)
            for task, sample in zip(tasks, batch_0):
                np.testing.assert_array_equal(answer(-1, *task)[1:], sample[1:])
            for sample in batch_1:
                np.testing.assert_array_equal(np.array([iter_worker_pid, i + 1]), sample)


@check_pool
def test_pool_many_ctxs(start_method):
    callbacks = [simple_callback, another_callback]
    groups = [MockGroup.from_callback(cb) for cb in callbacks]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=1, start_method=start_method
    ) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(0, 0, 0, 0),)]
        work_batch = TaskArgs.make_sample(SampleRange(0, 1, 0, 0))
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        pool.schedule_batch(context_i=1, work_batch=work_batch)
        batch_0 = pool.receive_batch(context_i=0)
        batch_1 = pool.receive_batch(context_i=1)
        for task, sample, pid in zip(tasks, batch_0, pids):
            np.testing.assert_array_equal(answer(pid, *task), sample)
        for task, sample, pid in zip(tasks, batch_1, pids):
            np.testing.assert_array_equal(answer(pid, *task) + 100, sample)


@check_pool
def test_pool_context_sync(start_method):
    callbacks = [simple_callback, another_callback]
    groups = [MockGroup.from_callback(cb, prefetch_queue_depth=3) for cb in callbacks]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=4, start_method=start_method
    ) as pool:
        capture_processes(pool)
        for i in range(4):
            tasks = [(SampleInfo(j, 0, 0, 0),) for j in range(10 * (i + 1))]
            work_batch = TaskArgs.make_sample(SampleRange(0, 10 * (i + 1), 0, 0))
            pool.schedule_batch(context_i=0, work_batch=work_batch)
            pool.schedule_batch(context_i=1, work_batch=work_batch)
        assert_scheduled_num(pool.contexts[0], 4)
        assert_scheduled_num(pool.contexts[1], 4)
        # pool after a reset should discard all previously scheduled tasks
        # (and sync workers to avoid race on writing to results buffer)
        pool.reset()
        tasks = [(SampleInfo(1000 + j, j, 0, 1),) for j in range(5)]
        work_batch = TaskArgs.make_sample(SampleRange(1000, 1005, 0, 1))
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        pool.schedule_batch(context_i=1, work_batch=work_batch)
        assert_scheduled_num(pool.contexts[0], 1)
        assert_scheduled_num(pool.contexts[1], 1)
        batch_0 = pool.receive_batch(context_i=0)
        batch_1 = pool.receive_batch(context_i=1)
        assert len(batch_0) == len(tasks)
        assert len(batch_1) == len(tasks)
        for task, sample in zip(tasks, batch_0):
            np.testing.assert_array_equal(answer(-1, *task)[1:], sample[1:])
        for task, sample in zip(tasks, batch_1):
            np.testing.assert_array_equal(answer(-1, *task)[1:] + 100, sample[1:])


@with_setup(setup_function, teardown_function)
def _test_multiple_stateful_sources_single_worker(num_workers):
    groups = [
        MockGroup.from_callback(IteratorCb(), batch=True),
        MockGroup.from_callback(IteratorCb(), batch=True),
    ]
    with create_pool(
        groups, keep_alive_queue_size=1, num_workers=num_workers, start_method="spawn"
    ) as pool:
        pids = get_pids(pool)
        assert len(pids) == min(num_workers, len(groups))
        pool.schedule_batch(context_i=0, work_batch=TaskArgs.make_batch((0,)))
        pool.schedule_batch(context_i=1, work_batch=TaskArgs.make_batch((0,)))
        iter_worker_num_0 = pool.contexts[0].dedicated_worker_id
        iter_worker_num_1 = pool.contexts[1].dedicated_worker_id
        iter_worker_pid_0 = pool.pool._processes[iter_worker_num_0].pid
        iter_worker_pid_1 = pool.pool._processes[iter_worker_num_1].pid
        batch_0 = pool.receive_batch(context_i=0)
        batch_1 = pool.receive_batch(context_i=1)
        np.testing.assert_array_equal(np.array([iter_worker_pid_0, 1]), batch_0[0])
        np.testing.assert_array_equal(np.array([iter_worker_pid_1, 1]), batch_1[0])
        if num_workers == 1:
            assert iter_worker_pid_0 == iter_worker_pid_1
        else:
            assert iter_worker_pid_0 != iter_worker_pid_1


def test_multiple_stateful_sources_single_worker():
    for num_workers in (1, 4):
        yield _test_multiple_stateful_sources_single_worker, num_workers


# ################################################################################################ #
# invalid return type
# ################################################################################################ #


def invalid_callback(i):
    return "42"


@raises(
    Exception,
    glob="Unsupported callback return type. Expected NumPy array, PyTorch or "
    "MXNet cpu tensors, DALI TensorCPU, or list or tuple of them representing sample. Got",
)
@with_setup(setup_function, teardown_function)
def test_pool_invalid_return():
    callbacks = [MockGroup.from_callback(invalid_callback)]
    with create_pool(
        callbacks, keep_alive_queue_size=1, num_workers=1, start_method="spawn"
    ) as pool:
        _ = get_pids(pool)
        work_batch = TaskArgs.make_sample(SampleRange(0, 1, 0, 0))
        pool.schedule_batch(context_i=0, work_batch=work_batch)
        pool.receive_batch(context_i=0)
