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

import os
import psutil
import weakref


def capture_processes(pool):
    """Need to be called to register the processes created by the pool. It is later used
    by the teardown_function to check if no process stayed alive after the test finished.
    """
    if pool is not None:
        pools.append(weakref.ref(pool))
        pool_processes.extend(pool.pids())
        proc_pool = pool.pool
        pool_threads.append(proc_pool._observer.thread)
    check_shm_for_dali("All shm chunks should be closed after initial pool setup, found {}")


def setup_function():
    """Prepare for the check if all started processes are no longer children of current process"""
    global pool_processes
    global pool_threads
    global pools
    pool_processes = []
    pool_threads = []
    pools = []


def teardown_function():
    """Check if there are no children processes started by the test after it ended.

    Be sure to call `capture_processes` in the test.
    """
    assert len(pool_processes), "No processes where tracked - did the test call capture_processes?"
    pools_not_collected = [pool_ref() is not None for pool_ref in pools]
    current_process = psutil.Process()
    children_pids = [process.pid for process in current_process.children()]
    left = set(pool_processes).intersection(children_pids)
    assert len(left) == 0, (
        f"Pipeline-started processes left after test is finished, pids alive: {left},\n"
        f"pids started during tests: {pool_processes}.\n"
        f"Pools not collected: {sum(pools_not_collected)}"
    )
    alive_threads = [thread.is_alive() for thread in pool_threads]
    assert sum(alive_threads) == 0, (
        "Some pool related threads are left after the test finished. "
        "Started in test suite: {}, still active: {}. "
        "Active threads map in the order of creation {}".format(
            len(pool_threads), sum(alive_threads), alive_threads
        )
    )


def check_shm_for_dali(msg):
    shm_paths = ["/dev/shm/", "/run/shm/"]
    for shm_path in shm_paths:
        if os.path.isdir(shm_path):
            shm_handles = os.listdir(shm_path)
            for handle in shm_handles:
                assert "nvidia_dali_" not in handle, msg.format(shm_path + handle)


def setup_module():
    check_shm_for_dali(
        "Expected clear shared mem environment before starting tests, "
        "found old DALI file handle: {}"
    )


def teardown_module():
    check_shm_for_dali("Test left opened shared memory file handle: {}")
