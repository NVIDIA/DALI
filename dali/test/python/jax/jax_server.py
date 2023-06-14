# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import jax
import numpy as np

import logging as log

from test_integration import get_dali_tensor_gpu

import nvidia.dali.plugin.jax as dax


def print_devices(process_id):
    log.info(f"Local devices = {jax.local_device_count()}, "
             f"global devices = {jax.device_count()}")

    log.info("All devices: ")
    print_devices_details(jax.devices(), process_id)

    log.info("Local devices:")
    print_devices_details(jax.local_devices(), process_id)


def print_devices_details(devices_list, process_id):
    for device in devices_list:
        log.info(f"Id = {device.id}, host_id = {device.host_id}, "
                 f"process_id = {device.process_index}, kind = {device.device_kind}")


def test_lax_workflow(process_id):
    array_from_dali = dax._to_jax_array(get_dali_tensor_gpu(1, (1), np.int32))

    assert array_from_dali.device() == jax.local_devices()[0], \
        "Array should be backed by the device local to current process."

    sum_across_devices = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(array_from_dali)

    assert sum_across_devices[0] == len(jax.devices()),\
        "Sum across devices should be equal to the number of devices as data per device = [1]"

    log.info("Passed lax workflow test")


def run_multiprocess_workflow(process_id=0):
    jax.distributed.initialize(
        coordinator_address="localhost:12321",
        num_processes=2,
        process_id=process_id)

    log.basicConfig(
        level=log.INFO,
        format=f"PID {process_id}: %(message)s")

    print_devices(process_id=process_id)
    test_lax_workflow(process_id=process_id)


if __name__ == "__main__":
    run_multiprocess_workflow(process_id=0)
