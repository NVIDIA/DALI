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

import nvidia.dali.plugin.jax as dax

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator

from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec, Mesh
from jax.experimental import mesh_utils
from utils import get_dali_tensor_gpu
import jax.numpy as jnp


def sequential_sharded_pipeline(
        batch_size, shape, device_id, shard_id, shard_size, multiple_outputs=False):
    """Helper to create DALI pipelines that return GPU tensors with sequential values
    and are iterating over virtual sharded dataset.

    For example setting shard_id for 2 and shard size for 8 will result in pipeline
    that starts its iteration from the sample with value 16 since this is third
    shard (shard_id=2) and the shard size is 8.

    Args:
        batch_size: Batch size for the pipeline.
        shape : Shape of the output tensor.
        device_id : Id of the device that pipeline will run on.
        shard_id : Id of the shard for the pipeline.
        shard_size : Size of the shard for the pipeline.
        multiple_outputs : If True, pipeline will return multiple outputs.
    """

    def create_numpy_sequential_tensors_callback():
        shard_offset = shard_size * shard_id

        def numpy_sequential_tensors(sample_info):
            return np.full(shape, sample_info.idx_in_epoch + shard_offset, dtype=np.int32)

        return numpy_sequential_tensors

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=device_id)
    def sequential_pipeline_def():
        data = fn.external_source(
            source=create_numpy_sequential_tensors_callback(),
            num_outputs=1,
            batch=False,
            dtype=types.INT32)
        data = data[0].gpu()

        if not multiple_outputs:
            return data

        return data, data + 0.25, data + 0.5

    return sequential_pipeline_def()


def test_dali_sequential_sharded_tensors_to_jax_sharded_array_manuall():
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    batch_size = 4
    shape = (1, 5)

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size, shape=shape, device_id=0, shard_id=0, shard_size=batch_size)
    pipe_0.build()

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size, shape=shape, device_id=1, shard_id=1, shard_size=batch_size)
    pipe_1.build()

    for batch_id in range(100):
        dali_tensor_gpu_0 = pipe_0.run()[0].as_tensor()
        dali_tensor_gpu_1 = pipe_1.run()[0].as_tensor()

        jax_shard_0 = dax.integration._to_jax_array(dali_tensor_gpu_0)
        jax_shard_1 = dax.integration._to_jax_array(dali_tensor_gpu_1)

        assert jax_shard_0.device() == jax.devices()[0]
        assert jax_shard_1.device() == jax.devices()[1]

        # when
        jax_array = jax.device_put_sharded(
            [jax_shard_0, jax_shard_1],
            [jax_shard_0.device(), jax_shard_1.device()])

        # then
        # Assert that all values are as expected
        # In first iteration, first shard should be:
        # [[0 0 0 0 0]
        #  [1 1 1 1 1]
        #  [2 2 2 2 2]
        #  [3 3 3 3 3]]
        # And second shrad should be:
        # [[4 4 4 4 4]
        #  [5 5 5 5 5]
        #  [6 6 6 6 6]
        #  [7 7 7 7 7]]
        # Then, in second iteration first shard should be:
        # [[4 4 4 4 4]
        #  [5 5 5 5 5]
        #  [6 6 6 6 6]
        #  [7 7 7 7 7]]
        # And second shard should be:
        # [[ 8  8  8  8  8]
        #  [ 9  9  9  9  9]
        #  [10 10 10 10 10]
        #  [11 11 11 11 11]]
        assert jax.numpy.array_equal(
            jax_array.device_buffers[0],
            jax.numpy.stack([
                jax.numpy.full(shape[1:], value, np.int32)
                for value in range(batch_id*batch_size, (batch_id+1)*batch_size)]))
        assert jax.numpy.array_equal(
            jax_array.device_buffers[1],
            jax.numpy.stack([
                jax.numpy.full(shape[1:], value, np.int32)
                for value in range((batch_id+1)*batch_size, (batch_id+2)*batch_size)]))

        # Assert correct backing devices for shards
        assert jax_array.device_buffers[0].device() == jax_shard_0.device()
        assert jax_array.device_buffers[1].device() == jax_shard_1.device()


def test_dali_sequential_sharded_tensors_to_jax_sharded_array_iterator_multiple_outputs():
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    batch_size = 4
    shape = (1, 5)

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=0,
        shard_id=0,
        shard_size=batch_size,
        multiple_outputs=True)

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=1,
        shard_id=1,
        shard_size=batch_size,
        multiple_outputs=True)

    output_names = ['data_0', 'data_1', 'data_2']

    # when
    dali_iterator = DALIGenericIterator([pipe_0, pipe_1], output_names, size=batch_size*10)

    for batch_id, batch in enumerate(dali_iterator):
        # then
        # check values for all outputs
        # for the data_0 values should be the same as in the single output example
        # for data_1 values are the same + 0.25, for data_2 the same + 0.5
        for output_id, output_name in enumerate(output_names):
            jax_array = batch[output_name]

            assert jax.numpy.array_equal(
                jax_array.device_buffers[0],
                jax.numpy.stack([
                    jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                    for value in range(batch_id*batch_size, (batch_id+1)*batch_size)]))
            assert jax.numpy.array_equal(
                jax_array.device_buffers[1],
                jax.numpy.stack([
                    jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                    for value in range((batch_id+1)*batch_size, (batch_id+2)*batch_size)]))

            # Assert correct backing devices for shards
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == 4


def run_sharding_test(sharding):
    # given
    dali_shard_0 = get_dali_tensor_gpu(0, (1), np.int32, 0)
    dali_shard_1 = get_dali_tensor_gpu(1, (1), np.int32, 1)

    shards = [dax.integration._to_jax_array(dali_shard_0),
              dax.integration._to_jax_array(dali_shard_1)]

    assert shards[0].device() == jax.devices()[0]
    assert shards[1].device() == jax.devices()[1]

    # when
    dali_sharded_array = jax.make_array_from_single_device_arrays(
        shape=(2,), sharding=sharding, arrays=shards)

    # then
    jax_sharded_array = jax.device_put(jnp.arange(2), sharding)

    assert (dali_sharded_array == jax_sharded_array).all()
    assert len(dali_sharded_array.device_buffers) == jax.device_count()

    assert dali_sharded_array.device_buffers[0].device() == jax.devices()[0]
    assert dali_sharded_array.device_buffers[1].device() == jax.devices()[1]


def run_sharding_iterator_test(sharding):
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    batch_size = 4
    shape = (1, 5)

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=0,
        shard_id=0,
        shard_size=batch_size,
        multiple_outputs=True)

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=1,
        shard_id=1,
        shard_size=batch_size,
        multiple_outputs=True)

    output_names = ['data_0', 'data_1', 'data_2']

    # when
    dali_iterator = DALIGenericIterator(
        [pipe_0, pipe_1], output_names, size=batch_size*10, sharding=sharding)

    for batch_id, batch in enumerate(dali_iterator):
        # then
        # check values for all outputs
        # for the data_0 values should be the same as in the single output example
        # for data_1 values are the same + 0.25, for data_2 the same + 0.5
        for output_id, output_name in enumerate(output_names):
            jax_array = batch[output_name]

            assert jax.numpy.array_equal(
                jax_array,
                jax.numpy.stack([
                    jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                    for value in range(batch_id*batch_size, (batch_id+2)*batch_size)]))

            # Assert correct backing devices for shards
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == 4


def test_positional_sharding_workflow():
    sharding = PositionalSharding(jax.devices())

    run_sharding_test(sharding)


def test_named_sharding_workflow():
    mesh = Mesh(jax.devices(), axis_names=('device'))
    sharding = NamedSharding(mesh, PartitionSpec('device'))

    run_sharding_test(sharding)


def test_positional_sharding_workflow_with_iterator():
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)

    run_sharding_iterator_test(sharding)


def test_named_sharding_workflow_with_iterator():
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    sharding = NamedSharding(mesh, PartitionSpec('batch'))

    run_sharding_iterator_test(sharding)
