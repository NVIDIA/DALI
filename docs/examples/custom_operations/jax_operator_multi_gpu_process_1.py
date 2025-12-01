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

# This is counterpart of the jax_operator_multi_gpu notebook,
# the notebook and this script run as a group,
# the notebook runs as process 0, this code runs as process 1


from functools import partial
import os

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

import nvidia.dali.fn as fn
from nvidia.dali.plugin.jax import data_iterator
from nvidia.dali.plugin.jax.fn import jax_function


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

jax.distributed.initialize(
    coordinator_address="localhost:12321",
    num_processes=2,
    process_id=1,
)

assert len(jax.devices()) == 2
assert len(jax.local_devices()) == 1

mesh = Mesh(jax.devices(), axis_names=("batch"))
sharding = NamedSharding(mesh, PartitionSpec("batch"))

dogs = [f"../data/images/dog/dog_{i}.jpg" for i in range(1, 9)]
kittens = [f"../data/images/kitten/cat_{i}.jpg" for i in range(1, 9)]


@data_iterator(
    output_map=["images"],
    sharding=sharding,
)
def iterator_function(shard_id, num_shards):
    assert num_shards == 2
    jpegs, _ = fn.readers.file(
        files=dogs if shard_id == 0 else kittens, name="image_reader"
    )
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, size=(244, 244))

    # mixup images between shards
    images = global_mixup(images)
    return images


@jax_function(sharding=sharding)
@jax.jit
@partial(
    shard_map,
    mesh=sharding.mesh,
    in_specs=PartitionSpec("batch"),
    out_specs=PartitionSpec("batch"),
)
@jax.vmap
def global_mixup(sample):
    mixed_up = 0.5 * sample + 0.5 * jax.lax.pshuffle(sample, "batch", [1, 0])
    mixed_up = jax.numpy.clip(mixed_up, 0, 255)
    return jax.numpy.array(mixed_up, dtype=jax.numpy.uint8)


local_batch_size = 8
num_shards = 2

iterator = iterator_function(
    batch_size=num_shards * local_batch_size, num_threads=4
)
batch = next(iterator)
