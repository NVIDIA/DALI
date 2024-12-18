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
import numpy as np

import nvidia.dali.plugin.jax as dax

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator, data_iterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec, Mesh
from jax.experimental import mesh_utils
from utils import get_dali_tensor_gpu, iterator_function_def
import jax.numpy as jnp

import itertools
from nose_utils import raises

# Common parameters for all tests in this file
batch_size = 4
shape = (1, 5)


def sequential_sharded_pipeline(
    batch_size, shape, device_id, shard_id, shard_size, multiple_outputs=False
):
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
            dtype=types.INT32,
        )
        data = data[0].gpu()

        if not multiple_outputs:
            return data

        return data, data + 0.25, data + 0.5

    return sequential_pipeline_def()


def test_dali_sequential_sharded_tensors_to_jax_sharded_array_manuall():
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size, shape=shape, device_id=0, shard_id=0, shard_size=batch_size
    )

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size, shape=shape, device_id=1, shard_id=1, shard_size=batch_size
    )

    for batch_id in range(100):
        dali_tensor_gpu_0 = pipe_0.run()[0].as_tensor()
        dali_tensor_gpu_1 = pipe_1.run()[0].as_tensor()

        jax_shard_0 = dax.integration._to_jax_array(dali_tensor_gpu_0, False)
        jax_shard_1 = dax.integration._to_jax_array(dali_tensor_gpu_1, False)

        assert jax_shard_0.device() == jax.devices()[0]
        assert jax_shard_1.device() == jax.devices()[1]

        # when
        jax_array = jax.device_put_sharded(
            [jax_shard_0, jax_shard_1], [jax_shard_0.device(), jax_shard_1.device()]
        )

        # then
        # Assert that all values are as expected
        # In first iteration, first shard should be:
        # [[0 0 0 0 0]
        #  [1 1 1 1 1]
        #  [2 2 2 2 2]
        #  [3 3 3 3 3]]
        # And second shard should be:
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
            jax.numpy.stack(
                [
                    jax.numpy.full(shape[1:], value, np.int32)
                    for value in range(batch_id * batch_size, (batch_id + 1) * batch_size)
                ]
            ),
        )
        assert jax.numpy.array_equal(
            jax_array.device_buffers[1],
            jax.numpy.stack(
                [
                    jax.numpy.full(shape[1:], value, np.int32)
                    for value in range((batch_id + 1) * batch_size, (batch_id + 2) * batch_size)
                ]
            ),
        )

        # Assert correct backing devices for shards
        assert jax_array.device_buffers[0].device() == jax_shard_0.device()
        assert jax_array.device_buffers[1].device() == jax_shard_1.device()


def test_dali_sequential_sharded_tensors_to_jax_sharded_array_iterator_multiple_outputs():
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=0,
        shard_id=0,
        shard_size=batch_size,
        multiple_outputs=True,
    )

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=1,
        shard_id=1,
        shard_size=batch_size,
        multiple_outputs=True,
    )

    output_names = ["data_0", "data_1", "data_2"]

    # when
    dali_iterator = DALIGenericIterator([pipe_0, pipe_1], output_names, size=batch_size * 10)

    for batch_id, batch in enumerate(dali_iterator):
        # then
        # check values for all outputs
        # for the data_0 values should be the same as in the single output example
        # for data_1 values are the same + 0.25, for data_2 the same + 0.5
        for output_id, output_name in enumerate(output_names):
            jax_array = batch[output_name]

            assert jax.numpy.array_equal(
                jax_array.device_buffers[0],
                jax.numpy.stack(
                    [
                        jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                        for value in range(batch_id * batch_size, (batch_id + 1) * batch_size)
                    ]
                ),
            )
            assert jax.numpy.array_equal(
                jax_array.device_buffers[1],
                jax.numpy.stack(
                    [
                        jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                        for value in range((batch_id + 1) * batch_size, (batch_id + 2) * batch_size)
                    ]
                ),
            )

            # Assert correct backing devices for shards
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == 4


def run_sharding_test(sharding):
    # given
    dali_shard_0 = get_dali_tensor_gpu(0, (1), np.int32, 0)
    dali_shard_1 = get_dali_tensor_gpu(1, (1), np.int32, 1)

    shards = [
        dax.integration._to_jax_array(dali_shard_0, False),
        dax.integration._to_jax_array(dali_shard_1, False),
    ]

    assert shards[0].device() == jax.devices()[0]
    assert shards[1].device() == jax.devices()[1]

    # when
    dali_sharded_array = jax.make_array_from_single_device_arrays(
        shape=(2,), sharding=sharding, arrays=shards
    )

    # then
    jax_sharded_array = jax.device_put(jnp.arange(2), sharding)

    assert (dali_sharded_array == jax_sharded_array).all()
    assert len(dali_sharded_array.device_buffers) == jax.device_count()

    assert dali_sharded_array.device_buffers[0].device() == jax.devices()[0]
    assert dali_sharded_array.device_buffers[1].device() == jax.devices()[1]


def run_sharding_iterator_test(sharding):
    assert jax.device_count() > 1, "Multigpu test requires more than one GPU"

    # given
    pipe_0 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=0,
        shard_id=0,
        shard_size=batch_size,
        multiple_outputs=True,
    )

    pipe_1 = sequential_sharded_pipeline(
        batch_size=batch_size,
        shape=shape,
        device_id=1,
        shard_id=1,
        shard_size=batch_size,
        multiple_outputs=True,
    )

    output_names = ["data_0", "data_1", "data_2"]

    # when
    dali_iterator = DALIGenericIterator(
        [pipe_0, pipe_1], output_names, size=batch_size * 10, sharding=sharding
    )

    for batch_id, batch in enumerate(dali_iterator):
        # then
        # check values for all outputs
        # for the data_0 values should be the same as in the single output example
        # for data_1 values are the same + 0.25, for data_2 the same + 0.5
        for output_id, output_name in enumerate(output_names):
            jax_array = batch[output_name]

            assert jax.numpy.array_equal(
                jax_array,
                jax.numpy.stack(
                    [
                        jax.numpy.full(shape[1:], value + output_id * 0.25, np.float32)
                        for value in range(batch_id * batch_size, (batch_id + 2) * batch_size)
                    ]
                ),
            )

            # Assert correct backing devices for shards
            assert jax_array.device_buffers[0].device() == jax.devices()[0]
            assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == 4


def test_positional_sharding_workflow():
    sharding = PositionalSharding(jax.devices())

    run_sharding_test(sharding)


def test_named_sharding_workflow():
    mesh = Mesh(jax.devices(), axis_names=("device"))
    sharding = NamedSharding(mesh, PartitionSpec("device"))

    run_sharding_test(sharding)


def test_positional_sharding_workflow_with_iterator():
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)

    run_sharding_iterator_test(sharding)


def test_named_sharding_workflow_with_iterator():
    mesh = Mesh(jax.devices(), axis_names=("batch"))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))

    run_sharding_iterator_test(sharding)


def run_sharded_iterator_test(iterator, num_iters=11):
    """Run the iterator with `sharding` set and assert that the output is as expected.

    Note: Output should be compatible with automatically parallelized functions. This means that
    the output should be an array, where slices of this array are shards of the output.
    """

    assert jax.device_count() == 2, "Sharded iterator test requires exactly 2 GPUs"

    batch_size_per_gpu = batch_size // jax.device_count()

    # Iterator should return 23 samples per shard
    assert iterator.size == 23

    # when
    for batch_id, batch in itertools.islice(enumerate(iterator), num_iters):
        # then
        jax_array = batch["tensor"]

        # For 2 GPUs expected result is as follows:
        # In first iteration, output should be:
        # [[0] [1] [23] [24]]
        # In first iteration, first shard should be:
        # [[0] [1]]
        # And second shard should be:
        # [[23] [24]]
        # Then, in the second iteration output should be:
        # [[2] [3] [25] [26]]
        # In second iteration first shard should be:
        # [[2] [3]]
        # And second shard should be:
        # [[25] [26]]
        sample_id = 0
        for device_id in range(jax.device_count()):
            for i in range(batch_size_per_gpu):
                ground_truth = jax.numpy.full(
                    (1), batch_id * batch_size_per_gpu + i + device_id * iterator.size, np.int32
                )
                assert jnp.array_equal(
                    jax_array[sample_id], ground_truth
                ), f"Expected {ground_truth} but got {jax_array[sample_id]}"
                sample_id += 1

        # Assert correct backing devices for shards
        assert jax_array.device_buffers[0].device() == jax.devices()[0]
        assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == num_iters - 1


def test_named_sharding_with_iterator_decorator():
    # given
    mesh = Mesh(jax.devices(), axis_names=("batch"))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))

    output_map = ["tensor"]

    # when
    @data_iterator(
        output_map=output_map,
        sharding=sharding,
        last_batch_policy=LastBatchPolicy.DROP,
        reader_name="reader",
    )
    def iterator_function(shard_id, num_shards):
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)

    data_iterator_instance = iterator_function(
        batch_size=batch_size,
        num_threads=4,
    )

    # then
    run_sharded_iterator_test(data_iterator_instance)


def test_positional_sharding_with_iterator_decorator():
    # given
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)

    output_map = ["tensor"]

    # when
    @data_iterator(
        output_map=output_map,
        sharding=sharding,
        last_batch_policy=LastBatchPolicy.DROP,
        reader_name="reader",
    )
    def iterator_function(shard_id, num_shards):
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)

    data_iterator_instance = iterator_function(batch_size=batch_size, num_threads=4)

    # then
    run_sharded_iterator_test(data_iterator_instance)


def test_dali_sequential_iterator_decorator_non_default_device():
    # given
    @data_iterator(output_map=["data"], reader_name="reader")
    def iterator_function():
        return iterator_function_def()

    # when
    iter = iterator_function(num_threads=4, device_id=1, batch_size=batch_size)

    batch = next(iter)

    # then
    assert batch["data"].device_buffers[0].device() == jax.devices()[1]


def run_pmapped_iterator_test(iterator, num_iters=11):
    """Run the iterator with `devices` set and assert that the output is as expected.

    Note: Output should be compatible with pmapped functions. This means that
    the output should be an array of arrays, where each array is a shard of the
    output.
    """

    assert jax.device_count() == 2, "Sharded iterator test requires exactly 2 GPUs"

    batch_size_per_gpu = batch_size // jax.device_count()

    # Iterator should return 23 samples per shard
    assert iterator.size == 23

    # when
    for batch_id, batch in itertools.islice(enumerate(iterator), num_iters):
        # then
        jax_array = batch["tensor"]

        # For 2 GPUs expected result is as follows:
        # In first iteration, output should be:
        # [[[ 0] [ 1]] [[23] [24]]]
        # In first iteration, first shard should be:
        # [[0] [1]]
        # And second shard should be:
        # [[23] [24]]
        # Then, in the second iteration output should be:
        # [[[ 2] [ 3]] [[25] [26]]]
        # In second iteration first shard should be:
        # [[2] [3]]
        # And second shard should be:
        # [[25] [26]]
        sample_id = 0
        for device_id in range(jax.device_count()):
            for i in range(batch_size_per_gpu):
                ground_truth = jax.numpy.full(
                    (1), batch_id * batch_size_per_gpu + i + device_id * iterator.size, np.int32
                )
                assert jnp.array_equal(
                    jax_array[device_id][i], ground_truth
                ), f"Expected {ground_truth} but got {jax_array[device_id][i]}"
                sample_id += 1

        # Assert correct backing devices for shards
        assert jax_array.device_buffers[0].device() == jax.devices()[0]
        assert jax_array.device_buffers[1].device() == jax.devices()[1]

    # Assert correct number of batches returned from the iterator
    assert batch_id == num_iters - 1


def test_iterator_decorator_with_devices():
    # given
    output_map = ["tensor"]

    # when
    @data_iterator(
        output_map=output_map,
        devices=jax.devices(),
        last_batch_policy=LastBatchPolicy.DROP,
        reader_name="reader",
    )
    def iterator_function(shard_id, num_shards):
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)

    data_iterator_instance = iterator_function(batch_size=batch_size, num_threads=4)

    # then
    run_pmapped_iterator_test(data_iterator_instance)


@raises(ValueError, glob="Only one of `sharding` and `devices` arguments can be provided.")
def test_sharding_and_devices_mutual_exclusivity():
    # given
    mesh = mesh_utils.create_device_mesh((jax.device_count(), 1))
    sharding = PositionalSharding(mesh)

    output_map = ["tensor"]

    # when
    @data_iterator(
        output_map=output_map, sharding=sharding, devices=jax.devices(), reader_name="reader"
    )
    def iterator_function(shard_id, num_shards):
        return iterator_function_def(shard_id=shard_id, num_shards=num_shards)
