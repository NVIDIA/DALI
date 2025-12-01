# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging as log
import argparse
from functools import partial


from nvidia.dali import fn
import nvidia.dali.plugin.jax as dax

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map


def print_devices_details(devices_list):
    for device in devices_list:
        log.info(
            f"Id = {device.id}, host_id = {device.host_id}, "
            f"process_id = {device.process_index}, kind = {device.device_kind}"
        )


def print_devices():
    log.info(
        f"Local devices = {jax.local_device_count()}, " f"global devices = {jax.device_count()}"
    )

    log.info("All devices: ")
    print_devices_details(jax.devices())

    log.info("Local devices:")
    print_devices_details(jax.local_devices())


def run_sharding_permute(cluster_size, proc_id):
    """Run DALI JAX iterator with a JAX function inside that permutes samples between shards"""

    mesh = Mesh(jax.devices(), axis_names=("shard"))
    sharding = NamedSharding(mesh, PartitionSpec("shard"))

    @dax.fn.jax_function(sharding=sharding)
    @jax.jit
    @partial(  # to be able to refer to "shard" axis with `pshuffle`
        shard_map,
        mesh=sharding.mesh,
        in_specs=(
            PartitionSpec("shard"),
            PartitionSpec("shard"),
        ),
        out_specs=PartitionSpec("shard"),
    )
    @jax.vmap  # vectorize across batch dimension
    def permute_between_shards(sample, sample_idx):
        should_swap = sample_idx % 2 == 1
        return jax.lax.cond(
            should_swap,
            lambda x: jax.lax.pshuffle(x, "shard", [1, 0]),
            lambda x: x,
            sample,
        )

    def sample_idx_source(sample_info):
        if sample_info.iteration >= 3:
            raise StopIteration
        return np.array(sample_info.idx_in_batch, dtype=np.int32)

    @dax.data_iterator(
        output_map=["data"],
        sharding=sharding,
    )
    def iterator_function(shard_id, num_shards):
        log.info(f"{shard_id}")
        sample_idx = fn.external_source(sample_idx_source, batch=False)
        # the shard_id will be the same in all samples in one process,
        # but different across processes.
        sample = fn.external_source(
            lambda sample_info: np.array(
                [[sample_info.idx_in_batch, shard_id, num_shards]], dtype=np.int32
            ),
            batch=False,
        )
        # permute the `sample` between processes, now every second sample
        # should be swapped.
        return permute_between_shards(sample.gpu(), sample_idx.gpu())

    batch_size = 7

    iterator = iterator_function(batch_size=cluster_size * batch_size, num_threads=4)
    local_devices = jax.local_devices()
    assert len(local_devices) == 1, f"{len(local_devices)}!= 1"
    local_device = local_devices[0]
    for out in iterator:
        data = out["data"]
        data_devices = list(data.devices())
        assert len(data_devices) == cluster_size, f"{len(data_devices)}!= {cluster_size}"
        local_arrays = [x.data for x in data.addressable_shards]
        assert len(local_arrays) == 1, f"{len(local_arrays)}!= 1"
        local_array = local_arrays[0]
        array_devices = list(local_array.devices())
        assert len(array_devices) == 1, f"{len(array_devices)}!= 1"
        array_device = array_devices[0]
        assert array_device == local_device, f"{array_device}!= {local_device}"
        ref = jax.numpy.array(
            [[[i, 1 - proc_id if i % 2 else proc_id, cluster_size]] for i in range(batch_size)]
        )
        assert jax.numpy.array_equal(local_array, ref), f"{local_array}!= {ref}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()

    cluster_size = 2
    proc_id = args.id
    assert 0 <= proc_id < cluster_size, f"{proc_id}, {cluster_size}"

    log.basicConfig(level=log.INFO, format=f"PID {(os.getpid(), args.id)}: %(message)s")
    jax.distributed.initialize(
        coordinator_address="localhost:12321", num_processes=cluster_size, process_id=proc_id
    )
    print_devices()
    assert cluster_size == len(jax.devices()), f"{cluster_size} != {len(jax.devices())}"

    run_sharding_permute(cluster_size=2, proc_id=args.id)
