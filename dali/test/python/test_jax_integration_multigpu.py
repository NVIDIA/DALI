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


def sequential_sharded_pipeline(batch_size, shape, device_id, shard_id, shard_size):
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
        return data

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

        jax_shard_0 = dax._to_jax_array(dali_tensor_gpu_0)
        jax_shard_1 = dax._to_jax_array(dali_tensor_gpu_1)

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
