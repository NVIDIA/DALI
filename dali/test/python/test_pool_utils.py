# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


def capture_processes(pool):
    """Need to be called to register the processes created by the pool. It is later used
    by the teardown_function to check if no process stayed alive after the test finished.
    """
    global pipe_processes
    if pool is not None:
        pipe_processes.update(pool.pids())


def setup_function():
    """Prepare for the check if all started processes are no longer children of current process
    """
    global pipe_processes
    pipe_processes = set()

def teardown_function():
    """Check if there are no children processes started by tests after it has ended.

    Be sure to call `capture_processes` in the test.
    """
    global pipe_processes
    current_process = psutil.Process()
    children = set(current_process.children())
    left = pipe_processes.intersection(children)
    assert len(left) == 0, ("Pipeline-started processes left after " +
          "test is finished, pids alive:\n{},\npids started during tests:\n{}").format(left, pipe_processes)

def check_shm_for_dali(msg):
    shm_paths = ["/dev/shm/", "/run/shm/"]
    for shm_path in shm_paths:
        if os.path.isdir(shm_path):
            shm_handles = os.listdir(shm_path)
            for handle in shm_handles:
                assert "nvidia_dali_" not in handle, msg.format(shm_path + handle)

def setup_module():
    check_shm_for_dali("Expected clear shared mem environment before starting tests, found old DALI file handle: {}")

def teardown_module():
    check_shm_for_dali("Test left opened shared memory file handle: {}")

