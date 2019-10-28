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
from shutil import rmtree as remove_directory

from nose import SkipTest
from nose.tools import raises, with_setup

try:
    tf.compat.v1.disable_eager_execution()
except:
    pass


class MnistPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, path, device, device_id = 0, shard_id = 0, num_shards = 1, seed = 0):
        super(MnistPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.device = device
        self.reader = ops.FileReader(file_root = path, random_shuffle = True)
        self.decode = ops.ImageDecoder(
            device = 'mixed' if device is 'gpu' else 'cpu', 
            output_type = types.GRAY)
        self.cmn = ops.CropMirrorNormalize(
            device = device,
            output_dtype = types.FLOAT,
            image_type = types.GRAY,
            mean = [0.],
            std = [255.],
            output_layout=types.NCHW)

    def define_graph(self):
        inputs, labels = self.reader(name = "Reader")
        images = self.decode(inputs)
        if self.device is 'gpu':
            labels = labels.gpu()
        images = self.cmn(images)

        return (images, labels)


def _get_mnist_dataset(batch_size, path, device = 'cpu', device_id = 0):
    mnist_pipeline = MnistPipeline(batch_size, 4, path, device, device_id)
    shapes = [
        (batch_size, 28, 28), 
        (batch_size)]
    dtypes = [
        tf.float32,
        tf.int32]

    return dali_tf.DALIDataset(
        pipeline=mnist_pipeline,
        batch_size= batch_size,
        shapes=shapes, 
        dtypes=dtypes,
        num_threads=4,
        device_id=device_id)


def _get_train_dataset(batch_size, device = 'cpu', device_id = 0):
    return _get_mnist_dataset(batch_size, '/data/MNIST/training', device, device_id)


def _get_test_dataset(batch_size, device = 'cpu', device_id = 0):
    return _get_mnist_dataset(
        batch_size, 
        '/data/MNIST/testing', 
        device, 
        device_id).take(batch_size)


def _keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def _test_keras_single_device(device = 'cpu', device_id = 0, batch_size = 32):
    skip_for_incompatible_tf()

    with tf.device('/{0}:{1}'.format(device, device_id)):
        model = _keras_model()

        train_dataset = _get_train_dataset(batch_size, device, device_id)

        model.fit(
            train_dataset,
            epochs = 5,
            steps_per_epoch = 100)

        assert model.evaluate(
            _get_test_dataset(batch_size, device, device_id))[1] > 0.9


def test_keras_single_gpu():
    _test_keras_single_device('gpu', 0)


def test_keras_single_other_gpu():
    _test_keras_single_device('gpu', 1)


def test_keras_single_cpu():
    _test_keras_single_device('cpu', 0)


def clear_checkpoints():
    remove_directory('/tmp/tensorflow-checkpoints', ignore_errors = True)


def _test_estimators_single_device(device = 'cpu', device_id = 0, batch_size = 32):
    skip_for_incompatible_tf()
    
    with tf.device('/{0}:{1}'.format(device, device_id)):
        def train_fn():
            return _get_train_dataset(batch_size, device, device_id).map(
                lambda features, labels: ({'x':features}, labels))

        feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

        classifier = tf.estimator.DNNClassifier(
            feature_columns = feature_columns,
            hidden_units = [128],
            n_classes = 10,
            dropout = 0.2,
            activation_fn=tf.nn.relu,
            model_dir = '/tmp/tensorflow-checkpoints',
            optimizer='Adam')

        classifier.train(input_fn = train_fn, steps = 500)

        def test_fn():
            return _get_test_dataset(batch_size, device, device_id).map(
                lambda features, labels: ({'x':features}, labels))

        assert classifier.evaluate(input_fn=test_fn)["accuracy"] > 0.8

@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_gpu():
    _test_estimators_single_device('gpu', 0)

@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_other_gpu():
    _test_estimators_single_device('gpu', 1)

@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_cpu():
    _test_estimators_single_device('cpu', 0)



    