# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax.numpy as jnp
import numpy as np
import logging as log
import argparse

from nvidia.dali import pipeline_def
from nvidia.dali.backend import TensorGPU
import nvidia.dali.types as types
import nvidia.dali.plugin.jax as dax

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import PositionalSharding
from jax.sharding import NamedSharding


def get_dali_tensor_gpu(value, shape, dtype, device_id=0) -> TensorGPU:
    """Helper function to create DALI TensorGPU.

    Args:
        value : Value to fill the tensor with.
        shape : Shape for the tensor.
        dtype : Data type for the tensor.

    Returns:
        TensorGPU: DALI TensorGPU with provided shape and dtype filled
        with provided value.
    """

    @pipeline_def(num_threads=1, batch_size=1)
    def dali_pipeline():
        values = types.Constant(value=np.full(shape, value, dtype), device="gpu")

        return values

    pipe = dali_pipeline(device_id=device_id)
    pipe.build()
    dali_output = pipe.run()

    return dali_output[0][0]


def print_devices(process_id):
    log.info(
        f"Local devices = {jax.local_device_count()}, " f"global devices = {jax.device_count()}"
    )

    log.info("All devices: ")
    print_devices_details(jax.devices(), process_id)

    log.info("Local devices:")
    print_devices_details(jax.local_devices(), process_id)


def print_devices_details(devices_list, process_id):
    for device in devices_list:
        log.info(
            f"Id = {device.id}, platform = {device.platform}, "
            f"process_id = {device.process_index}, kind = {device.device_kind}"
        )


def run_distributed_sharing_test(sharding, process_id):
    log.info(f"Sharding: {sharding}")

    dali_local_shards = []
    for id, device in enumerate(jax.local_devices()):
        current_shard = dax.integration._to_jax_array(
            get_dali_tensor_gpu(process_id, (1), np.int32, id), False
        )

        assert dax.integration._jax_device(current_shard) == device

        dali_local_shards.append(current_shard)

    dali_sharded_array = jax.make_array_from_single_device_arrays(
        shape=(jax.device_count(),), sharding=sharding, arrays=dali_local_shards
    )

    assert len(dali_sharded_array.device_buffers) == jax.local_device_count()

    for id, buffer in enumerate(dali_sharded_array.device_buffers):
        assert buffer == jnp.array([process_id])
        assert dax.integration._jax_device(buffer) == jax.local_devices()[id]


def test_positional_sharding_workflow(process_id):
    sharding = PositionalSharding(jax.devices())

    run_distributed_sharing_test(sharding=sharding, process_id=process_id)

    log.info("Passed positional sharding workflow test")


def test_named_sharding_workflow(process_id):
    mesh = Mesh(jax.devices(), axis_names=("device"))
    sharding = NamedSharding(mesh, PartitionSpec("device"))

    run_distributed_sharing_test(sharding=sharding, process_id=process_id)

    log.info("Passed named sharding workflow test")


def run_multiprocess_workflow(process_id=0, cluster_size=1):
    jax.distributed.initialize(
        coordinator_address="localhost:12321", num_processes=cluster_size, process_id=process_id
    )

    log.basicConfig(format=f"PID {process_id}: %(message)s", level=log.INFO)

    print_devices(process_id=process_id)

    test_positional_sharding_workflow(process_id=process_id)
    test_named_sharding_workflow(process_id=process_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()

    run_multiprocess_workflow(process_id=0, cluster_size=args.size)
