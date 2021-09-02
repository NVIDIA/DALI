# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._multiproc.pool import WorkerPool, ProcPool
from contextlib import closing
from nvidia.dali.types import SampleInfo
from functools import wraps
import numpy as np
import os
from nose.tools import with_setup
from nose_utils import raises

from test_pool_utils import *

def answer(pid, info):
    return np.array([pid, info.idx_in_epoch, info.idx_in_batch, info.iteration])

def simple_callback(info):
    pid = os.getpid()
    return answer(pid, info)

def create_pool(callbacks, queue_depth=1, num_workers=1, start_method="fork"):
    queue_depths = [queue_depth for _ in callbacks]
    proc_pool = None
    try:
        proc_pool = ProcPool(callbacks, queue_depths, num_workers=num_workers,
                            start_method=start_method, initial_chunk_size=1024 * 1024)
        worker_pool = WorkerPool(len(callbacks), queue_depths, proc_pool)
        capture_processes(proc_pool)
        return closing(worker_pool)
    except:
        proc_pool.close()
        raise

def get_pids(worker_pool):
    # Note that we also capture the pids so the setup_function and teardown_function can
    # verify its correctness.
    return worker_pool.pids()

start_methods=["fork", "spawn"]

# Invoke the `fn` with all start methods. Call setup and teardown before and after the test.
#
# We do this to not repeat the pattern of:
#
# def check_somthing(start_method):
#    ...
#
# @with_setup(setup_function, teardown_function)
# def test_something():
#   for start_method in start_methods:
#      yield check_somthing, start_method
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
    callbacks = [simple_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=1, start_method=start_method) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(0, 0, 0, 0),)]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        batch = pool.receive_batch(context_i=0)
        for task, sample in zip(tasks, batch):
            np.testing.assert_array_equal(answer(pid, *task), sample)


@check_pool
def test_pool_multi_task(start_method):
    callbacks = [simple_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=1, start_method=start_method) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(i, i, 0, 0),) for i in range(10)]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        batch = pool.receive_batch(context_i=0)
        for task, sample in zip(tasks, batch):
            np.testing.assert_array_equal(answer(pid, *task), sample)


# Test that we can hold as many results as the queue depth and assert collected batches
# are propely cleaned up from the context
@check_pool
def test_pool_scheduled_tasks_cleanup(start_method):
    callbacks = [simple_callback]
    for depth in [1, 2, 4, 8]:
        with create_pool(callbacks, queue_depth=depth, num_workers=1, start_method=start_method) as pool:
            pids = get_pids(pool)
            pid = pids[0]
            task_list = [(i, [(SampleInfo(i, 0, i, 0),)]) for i in range(depth)]
            for i, tasks in task_list:
                pool.schedule_batch(context_i=0, dst_chunk_i=i, tasks=tasks)
            assert len(pool.contexts[0].partially_received) == depth
            batches = []
            for i in range(depth):
                batches.append(pool.receive_batch(context_i=0))
                assert len(pool.contexts[0].partially_received) == depth - 1 - i
            tasks_batches = zip(task_list, batches)
            for (i, tasks), batch in tasks_batches:
                for task, sample in zip(tasks, batch):
                    np.testing.assert_array_equal(answer(pid, *task), sample)


# ################################################################################################ #
# 1 callback, multiple workers tests
# ################################################################################################ #


@check_pool
def test_pool_work_split_2_tasks(start_method):
    callbacks = [simple_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=2, start_method=start_method) as pool:
        pids = get_pids(pool)
        tasks = [(SampleInfo(0, 0, 0, 0),), (SampleInfo(1, 1, 0, 0),)]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        batch = pool.receive_batch(context_i=0)
        for task, sample, pid in zip(tasks, batch, pids):
            np.testing.assert_array_equal(answer(pid, *task), sample)


@check_pool
def test_pool_work_split_multiple_tasks(start_method):
    callbacks = [simple_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=2, start_method=start_method) as pool:
        num_tasks = 16
        pids = get_pids(pool)
        tasks = [(SampleInfo(i, i, 0, 0),) for i in range(num_tasks)]
        split_pids = []
        assert num_tasks % len(pids) == 0, "Testing only even splits"
        for pid in pids:
            split_pids += [pid] * (num_tasks // len(pids))
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        batch = pool.receive_batch(context_i=0)
        for task, sample, pid in zip(tasks, batch, split_pids):
            np.testing.assert_array_equal(answer(pid, *task), sample)


# ################################################################################################ #
# multiple callback, 1 worker tests
# ################################################################################################ #


def another_callback(info):
    return simple_callback(info) + 100


@check_pool
def test_pool_many_ctxs(start_method):
    callbacks = [simple_callback, another_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=1, start_method=start_method) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(0, 0, 0, 0),)]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        pool.schedule_batch(context_i=1, dst_chunk_i=0, tasks=tasks)
        batch_0 = pool.receive_batch(context_i=0)
        batch_1 = pool.receive_batch(context_i=1)
        for task, sample, pid in zip(tasks, batch_0, pids):
            np.testing.assert_array_equal(answer(pid, *task), sample)
        for task, sample, pid in zip(tasks, batch_1, pids):
            np.testing.assert_array_equal(answer(pid, *task) + 100, sample)


# Check that the same worker executes the ctxs
@check_pool
def test_pool_many_ctxs_many_workers(start_method):
    callbacks = [simple_callback, another_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=5, start_method=start_method) as pool:
        pids = get_pids(pool)
        pid = pids[0]
        tasks = [(SampleInfo(0, 0, 0, 0),)]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        pool.schedule_batch(context_i=1, dst_chunk_i=0, tasks=tasks)
        batch_0 = pool.receive_batch(context_i=0)
        batch_1 = pool.receive_batch(context_i=1)
        for task, sample, pid in zip(tasks, batch_0, pids):
            np.testing.assert_array_equal(answer(pid, *task), sample)
        for task, sample, pid in zip(tasks, batch_1, pids):
            np.testing.assert_array_equal(answer(pid, *task) + 100, sample)

# ################################################################################################ #
# invalid return type
# ################################################################################################ #


def invalid_callback():
    return "42"

@raises(Exception, glob="Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, DALI TensorCPU, or list or tuple of them representing sample. Got")
@with_setup(setup_function, teardown_function)
def test_pool_invalid_return():
    callbacks = [invalid_callback]
    with create_pool(callbacks, queue_depth=1, num_workers=1, start_method="spawn") as pool:
        tasks = [()]
        pool.schedule_batch(context_i=0, dst_chunk_i=0, tasks=tasks)
        pool.receive_batch(context_i=0)
