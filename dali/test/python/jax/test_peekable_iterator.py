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


import jax.numpy as jnp
from nvidia.dali.plugin.jax.clu import DALIGenericPeekableIterator as DALIIteraor
from test_integration import sequential_pipeline
from clu.data.dataset_iterator import ArraySpec
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import numpy as np
from nose_utils import raises


def pipeline_with_variable_shape_output(batch_size):
    """Helper to create DALI pipelines that return GPU tensors with variable shape.

    Args:
        batch_size: Batch size for the pipeline.
    """

    def numpy_tensors(sample_info):
        tensors = [
            np.full((1,5), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1,3), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1,2), sample_info.idx_in_epoch, dtype=np.int32),
            np.full((1,4), sample_info.idx_in_epoch, dtype=np.int32),
        ]
        return tensors[sample_info.idx_in_epoch % len(tensors)]

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def sequential_pipeline_def():
        data = fn.external_source(
            source=numpy_tensors,
            num_outputs=1,
            batch=False,
            dtype=types.INT32)
        data = data[0].gpu()
        return data

    return sequential_pipeline_def()

import os

training_data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')

def mnist_pipeline(batch_size):
    image_size = 28

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=1, seed=0)
    def mnist_pipeline():
        jpegs, labels = fn.readers.caffe2(
            path=training_data_path,
            random_shuffle=False,
            name="mnist_caffe2_reader")
        images = fn.decoders.image(
            jpegs, device='cpu', output_type=types.GRAY)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, std=[255.], output_layout="CHW")
        images = fn.reshape(images, shape=[image_size * image_size])

        return images
    
    return mnist_pipeline()


def test_jax_peekable_iterator_peek():
    # given
    batch_size = 3
    shape = (1, 5)
    batch_shape = (batch_size, *shape[1:])
    pipe = sequential_pipeline(batch_size, shape)
    
    # when
    iterator = DALIIteraor([pipe], ['data'], size=batch_size*100)
    
    # then
    assert iterator.element_spec == { 'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}
    
    for i in range(10):
        peeked_output = iterator.peek()
        output = iterator.next()
        
        assert jnp.array_equal(
            output['data'], peeked_output['data'])
    
    
def test_jax_peekable_iterator_peek_async():
    # given
    batch_size = 3
    shape = (1, 5)
    batch_shape = (batch_size, *shape[1:])
    pipe = sequential_pipeline(batch_size, shape)
    
    # when
    iterator = DALIIteraor([pipe], ['data'], size=batch_size*100)
    
    # then
    assert iterator.element_spec == { 'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}
    
    for i in range(10):
        print("iteration", i, "===================")
        print("call peek_async")
        peeked_output = iterator.peek_async()
        peeked_output_1 = iterator.peek_async()
        peeked_output_1.result()
        
        print("call next")
        output = iterator.next()
        
        peeked_output = peeked_output.result()
        
        # peeked_output = peeked_output.result()
        
        # assert jnp.array_equal(
        #     output['data'], peeked_output.result()['data']), \
        #     f"output: {output['data']}, peeked_output: {peeked_output.result()['data']}"
            
        assert jnp.array_equal(
            output['data'], peeked_output['data']), \
            f"output: {output['data']}, peeked_output: {peeked_output['data']}"
            
        print()


import tensorflow as tf
from clu.data.dataset_iterator import TfDatasetIterator


def create_tf_iterator(start_index: int = 0, checkpoint: bool = False):
    primes = tf.constant([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    ds = tf.data.Dataset.range(start_index, 10)

    ds = ds.map(lambda i: {"_index": i, "prime": primes[i]})
    # Remove index 1 and 3.
    ds = ds.filter(lambda x: tf.logical_and(x["prime"] != 3, x["prime"] != 7))
    ds = ds.batch(2, drop_remainder=True)
    return TfDatasetIterator(ds, checkpoint=checkpoint)


def test_jax_peekable_tf():
    iterator = create_tf_iterator()

    # then
    # assert iterator.element_spec == { 'data': ArraySpec(dtype=jnp.int32, shape=3)}
    
    for i in range(10):
        print("iteration", i, "===================")
        print("call peek_async")
        peeked_output = iterator.peek_async()
        peeked_output = peeked_output.result()
        
        print("call next")
        output = iterator.next()
        
        # peeked_output = peeked_output.result()
        
        # assert jnp.array_equal(
        #     output['data'], peeked_output.result()['data']), \
        #     f"output: {output['data']}, peeked_output: {peeked_output.result()['data']}"
            
        assert jnp.array_equal(
            output['data'], peeked_output['data']), \
            f"output: {output['data']}, peeked_output: {peeked_output['data']}"
            
        print()
    

@raises(ValueError, glob="The shape or type of the output changed between iterations.")
def test_jax_peekable_iterator_with_variable_shapes_pipeline():
    # given
    batch_size = 1
    pipe = pipeline_with_variable_shape_output(batch_size)
    
    iterator = DALIIteraor([pipe], ['data'], size=batch_size*100)
    iterator.next()
    
    # when
    iterator.next()
