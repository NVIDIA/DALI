# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import nvidia.dali as dali
import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
import os
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from nose.tools import raises

test_data_root = os.environ['DALI_EXTRA_PATH']
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id = 0, seed = 0):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            shard_id = 0, 
            num_shards = 1, 
            ratio=False, 
            save_img_ids=True)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.resize = ops.Resize(
            device = "cpu",
            image_type = types.RGB,
            interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(
            device = "cpu",
            output_dtype = types.FLOAT,
            crop = (224, 224),
            image_type = types.RGB,
            mean = [128., 128., 128.],
            std = [1., 1., 1.])
        self.res_uniform = ops.Uniform(range = (256.,480.))
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.cast = ops.Cast(
            device = "cpu",
            dtype = types.FLOAT)

    def define_graph(self):
        inputs, _, _, im_ids = self.input()
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.res_uniform())
        output = self.cmn(
            images, 
            crop_pos_x = self.uniform(),
            crop_pos_y = self.uniform())
        im_ids_float = self.cast(im_ids)

        return (
            output, 
            im_ids,
            im_ids_float)


def test_tf_dataset():
    batch_size = 12
    num_threads = 4
    epochs = 10

    dataset_pipeline = TestPipeline(batch_size, num_threads)
    shapes = [
        (batch_size, 3, 224, 224), 
        (batch_size, 1),
        (batch_size, 1)]
    dtypes = [
        tf.float32,
        tf.int32, 
        tf.float32]

    dataset_results = []
    with tf.device('/cpu:0'):
        daliset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            shapes=shapes, 
            dtypes=dtypes,
            num_threads=num_threads)

        with tf.compat.v1.Session() as sess:
            iterator = tf.compat.v1.data.make_one_shot_iterator(daliset)
            next_element = iterator.get_next()
            for _ in range(epochs):
                dataset_results.append(sess.run(next_element))

    standalone_pipeline = TestPipeline(batch_size, num_threads)
    standalone_pipeline.build()
    standalone_results = []
    for _ in range(epochs):
        standalone_results.append(
            tuple(result.as_array() for result in standalone_pipeline.run()))
        
    for dataset_result, standalone_result in zip(dataset_results, standalone_results):
        for dataset_out, standalone_out in zip(dataset_result, standalone_result):
            assert np.array_equal(dataset_out, standalone_out)

@raises(Exception)
def test_differnt_num_shapes_dtypes():
    batch_size = 12
    num_threads = 4

    dataset_pipeline = TestPipeline(batch_size, num_threads)
    shapes = [
        (batch_size, 3, 224, 224), 
        (batch_size, 1),
        (batch_size, 1)]
    dtypes = [
        tf.float32,
        tf.float32]

    with tf.device('/cpu:0'):
        dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            shapes=shapes, 
            dtypes=dtypes,
            num_threads=num_threads)
