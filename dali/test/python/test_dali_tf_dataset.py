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

from nose import SkipTest
from nose.tools import raises

test_data_root = os.environ['DALI_EXTRA_PATH']
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

def skip_for_incompatible_tf():
    if tf.__version__.split('.')[1] not in {'13', '14'}:
        raise SkipTest('This feature is enabled for TF 1.13 and 1.14 only')


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device, device_id = 0, seed = 0):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.device = device
        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            shard_id = 0, 
            num_shards = 1, 
            ratio=False, 
            save_img_ids=True)
        self.decode = ops.ImageDecoder(
            device = 'mixed' if device is 'gpu' else 'cpu', 
            output_type = types.RGB)
        self.resize = ops.Resize(
            device = device,
            image_type = types.RGB,
            interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(
            device = device,
            output_dtype = types.FLOAT,
            crop = (224, 224),
            image_type = types.RGB,
            mean = [128., 128., 128.],
            std = [1., 1., 1.])
        self.res_uniform = ops.Uniform(range = (256.,480.))
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.cast = ops.Cast(
            device = device,
            dtype = types.INT16)

    def define_graph(self):
        inputs, _, _, im_ids = self.input()
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.res_uniform())
        output = self.cmn(
            images, 
            crop_pos_x = self.uniform(),
            crop_pos_y = self.uniform())
        if self.device is 'gpu':
            im_ids = im_ids.gpu()
        im_ids_16 = self.cast(im_ids)

        return (
            output, 
            im_ids,
            im_ids_16)


def _dataset_options():
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.autotune = False
    except:
        print('Could not set TF Dataset Options')
        options.experimental_autotune = False

    return options


def _test_tf_dataset(device):
    skip_for_incompatible_tf()

    batch_size = 12
    num_threads = 4
    epochs = 10

    dataset_pipeline = TestPipeline(batch_size, num_threads, device)
    shapes = [
        (batch_size, 3, 224, 224), 
        (batch_size, 1),
        (batch_size, 1)]
    dtypes = [
        tf.float32,
        tf.int32, 
        tf.int16]

    dataset_results = []
    with tf.device('/{0}:0'.format(device)):
        daliset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            shapes=shapes, 
            dtypes=dtypes,
            num_threads=num_threads)
        daliset = daliset.with_options(_dataset_options())

        iterator = tf.compat.v1.data.make_initializable_iterator(daliset)
        next_element = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(iterator.initializer)
            for _ in range(epochs):
                dataset_results.append(sess.run(next_element))

    standalone_pipeline = TestPipeline(batch_size, num_threads, device)
    standalone_pipeline.build()
    standalone_results = []
    for _ in range(epochs):
        if device is 'gpu':
            standalone_results.append(
                tuple(result.as_cpu().as_array() for result in standalone_pipeline.run()))
        else:
            standalone_results.append(
                tuple(result.as_array() for result in standalone_pipeline.run()))
        
    for dataset_result, standalone_result in zip(dataset_results, standalone_results):
        for dataset_out, standalone_out in zip(dataset_result, standalone_result):
            assert np.array_equal(dataset_out, standalone_out)

def test_tf_dataset_gpu():
    _test_tf_dataset('gpu')


def test_tf_dataset_cpu():
    _test_tf_dataset('cpu')


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
