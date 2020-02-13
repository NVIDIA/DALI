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
from tensorflow.python.client import device_lib
import nvidia.dali.plugin.tf as dali_tf
import os
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from distutils.version import StrictVersion
from test_utils_tensorflow import *

from nose import SkipTest
from nose.tools import raises

try:
    tf.compat.v1.disable_eager_execution()
except:
    pass

test_data_root = os.environ['DALI_EXTRA_PATH']
file_root = os.path.join(test_data_root, 'db', 'coco_dummy', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco_dummy', 'instances.json')


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device, device_id = 0, shard_id = 0, num_shards = 1, seed = 0):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.device = device
        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            shard_id = shard_id,
            num_shards = num_shards,
            ratio=False,
            save_img_ids=True)
        self.decode = ops.ImageDecoder(
            device = 'mixed' if device is 'gpu' else 'cpu',
            output_type = types.RGB)
        self.resize = ops.Resize(
            device = device,
            image_type = types.RGB,
            resize_x = 224,
            resize_y = 224,
            interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(
            device = device,
            output_dtype = types.FLOAT,
            image_type = types.RGB,
            mean = [128., 128., 128.],
            std = [1., 1., 1.])
        self.cast = ops.Cast(
            device = device,
            dtype = types.INT16)


    def define_graph(self):
        inputs, _, _, im_ids = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.resize(images)
        output = self.cmn(
            images)
        if self.device is 'gpu':
            im_ids = im_ids.gpu()
        im_ids_16 = self.cast(im_ids)

        return (
            output,
            im_ids,
            im_ids_16)


def setup():
    skip_for_incompatible_tf()


def _test_tf_dataset(device, device_id = 0):
    batch_size = 12
    num_threads = 4
    iterations = 10

    dataset_pipeline = TestPipeline(batch_size, num_threads, device, device_id)
    shapes = [
        (batch_size, 3, 224, 224),
        (batch_size, 1),
        (batch_size, 1)]
    dtypes = [
        tf.float32,
        tf.int32,
        tf.int16]

    dataset_results = []
    with tf.device('/{0}:{1}'.format(device, device_id)):
        daliset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            shapes=shapes,
            dtypes=dtypes,
            num_threads=num_threads,
            device_id=device_id)
        daliset = daliset.with_options(dataset_options())

        iterator = tf.compat.v1.data.make_initializable_iterator(daliset)
        next_element = iterator.get_next()

    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), iterator.initializer])
        for _ in range(iterations):
            dataset_results.append(sess.run(next_element))

    standalone_pipeline = TestPipeline(batch_size, num_threads, device, device_id = 0)
    standalone_pipeline.build()
    standalone_results = []
    for _ in range(iterations):
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


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    _test_tf_dataset('gpu', 1)


def test_tf_dataset_cpu():
    _test_tf_dataset('cpu')


@raises(Exception)
def test_different_num_shapes_dtypes():
    batch_size = 12
    num_threads = 4

    dataset_pipeline = TestPipeline(batch_size, num_threads, 'cpu')
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


def _test_tf_dataset_multigpu():
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4

    shapes = [
        (batch_size, 3, 224, 224),
        (batch_size, 1),
        (batch_size, 1)]
    dtypes = [
        tf.float32,
        tf.int32,
        tf.int16]

    dataset_results = []
    initializers = [tf.compat.v1.global_variables_initializer()]
    ops_to_run = []

    for device_id in range(num_devices):
        with tf.device('/gpu:{0}'.format(device_id)):
            dataset_pipeline = TestPipeline(batch_size, num_threads, 'gpu', device_id, device_id, num_devices)
            daliset = dali_tf.DALIDataset(
                pipeline=dataset_pipeline,
                batch_size=batch_size,
                shapes=shapes,
                dtypes=dtypes,
                num_threads=num_threads,
                device_id=device_id)
            daliset = daliset.with_options(dataset_options())

            iterator = tf.compat.v1.data.make_initializable_iterator(daliset)
            initializers.append(iterator.initializer)

            ops_to_run.append(iterator.get_next())


    standalone_pipeline = TestPipeline(
        batch_size, num_threads, device = 'gpu', device_id = 0)
    standalone_pipeline.build()

    dataset_size = standalone_pipeline.epoch_size("Reader")
    iterations = dataset_size // batch_size

    with tf.compat.v1.Session() as sess:
        sess.run(initializers)
        for _ in range(iterations):
            dataset_results.append(sess.run(ops_to_run))

    standalone_results = []
    for _ in range(iterations):
        standalone_results.append(
            tuple(result.as_cpu().as_array() for result in standalone_pipeline.run()))

    assert len(dataset_results) == iterations
    for it in range(iterations):
        assert len(dataset_results[it]) == num_devices
        for device_id in range(num_devices):
            batch_id = iterations - ((it + device_id * (iterations // num_devices)) % iterations) - 1
            it_id = iterations - it - 1
            assert np.array_equal(
                standalone_results[it_id][0],
                dataset_results[batch_id][device_id][0])
            assert np.array_equal(
                standalone_results[it_id][1],
                dataset_results[batch_id][device_id][1])
            assert np.array_equal(
                standalone_results[it_id][2],
                dataset_results[batch_id][device_id][2])


class PythonOperatorPipeline(Pipeline):
    def __init__(self):
        super(PythonOperatorPipeline, self).__init__(1, 1, 0, 0)
        self.python_op = ops.PythonFunction(function=lambda: np.zeros((3, 3, 3)))

    def define_graph(self):
        return self.python_op()


@raises(RuntimeError)
def test_python_operator_error():
    dataset_pipeline = PythonOperatorPipeline()
    shapes = [(1, 3, 3, 3)]
    dtypes = [tf.float32]

    with tf.device('/cpu:0'):
        daliset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=1,
            shapes=shapes,
            dtypes=dtypes,
            num_threads=1,
            device_id=0)
