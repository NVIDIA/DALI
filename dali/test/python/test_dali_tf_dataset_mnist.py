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
from test_utils_tensorflow import *
from shutil import rmtree as remove_directory

from nose.tools import with_setup

try:
    from tensorflow.compat.v1 import Session
    from tensorflow.compat.v1 import placeholder
    from tensorflow.compat.v1 import truncated_normal
except:
    # Older TF versions don't have compat.v1 layer
    from tensorflow import Session
    from tensorflow import placeholder
    from tensorflow import truncated_normal

try:
    from tensorflow.train import AdamOptimizer as Adam
    from tensorflow.train import AdamOptimizer as AdamOptimizer
except:
    from tensorflow.compat.v1.keras.optimizers import Adam
    from tensorflow.compat.v1.train import AdamOptimizer

try:
    tf.compat.v1.disable_eager_execution()
except:
    pass

target = 0.9
batch_size = 32
dropout = 0.2
image_size = 28
labels_size = 10
hidden_size = 128
epochs = 5
iterations = 100


class MnistPipeline(Pipeline):
    def __init__(self, num_threads, path, device, device_id=0, shard_id=0, num_shards=1, seed=0):
        super(MnistPipeline, self).__init__(
            batch_size, num_threads, device_id, seed)
        self.device = device
        self.reader = ops.FileReader(file_root=path, random_shuffle=True)
        self.decode = ops.ImageDecoder(
            device='mixed' if device is 'gpu' else 'cpu',
            output_type=types.GRAY)
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            output_dtype=types.FLOAT,
            image_type=types.GRAY,
            mean=[0.],
            std=[255.],
            output_layout=types.NCHW)

    def define_graph(self):
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)
        if self.device is 'gpu':
            labels = labels.gpu()
        images = self.cmn(images)

        return (images, labels)


def _get_mnist_dataset(path, device='cpu', device_id=0):
    mnist_pipeline = MnistPipeline(4, path, device, device_id)
    shapes = [
        (batch_size, image_size, image_size),
        (batch_size)]
    dtypes = [
        tf.float32,
        tf.int32]

    daliset = dali_tf.DALIDataset(
        pipeline=mnist_pipeline,
        batch_size=batch_size,
        shapes=shapes,
        dtypes=dtypes,
        num_threads=4,
        device_id=device_id)
    return daliset.with_options(dataset_options())


def _get_train_dataset(device='cpu', device_id=0):
    return _get_mnist_dataset(
        '/data/MNIST/training',
        device,
        device_id)


def _test_graph_single_device(device='cpu', device_id=0):
    skip_for_incompatible_tf()

    with tf.device('/{0}:{1}'.format(device, device_id)):
        daliset = _get_train_dataset(device, device_id)

        iterator = tf.compat.v1.data.make_initializable_iterator(daliset)
        images, labels = iterator.get_next()
        keep_prob = placeholder(tf.float32)

        images = tf.reshape(images, [batch_size, image_size*image_size])
        labels = tf.reshape(
            tf.one_hot(labels, labels_size),
            [batch_size, labels_size])
        W_h = tf.Variable(truncated_normal(
            [image_size*image_size, hidden_size], stddev=0.1))
        b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

        hidden = tf.nn.relu(tf.matmul(images, W_h) + b_h)

        W = tf.Variable(truncated_normal(
            [hidden_size, labels_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

        output = tf.matmul(hidden, W) + b

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=output))

        train_step = AdamOptimizer().minimize(loss)
        correct_prediction = tf.equal(
            tf.argmax(output, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with Session() as sess:
        sess.run(
            [tf.compat.v1.global_variables_initializer(), iterator.initializer])
        for i in range(epochs * iterations):
            sess.run(train_step, feed_dict={keep_prob: dropout})

            if i % iterations == 0:
                train_accuracy = accuracy.eval(feed_dict={keep_prob: 1.0})
                print("Step %d, accuracy: %g" % (i, train_accuracy))

        final_accuracy = 0
        for _ in range(iterations):
            final_accuracy = final_accuracy + \
                accuracy.eval(feed_dict={keep_prob: 1.0})
        final_accuracy = final_accuracy / iterations

        assert final_accuracy > target


def test_graph_single_gpu():
    _test_graph_single_device('gpu', 0)


def test_graph_single_other_gpu():
    _test_graph_single_device('gpu', 1)


def test_graph_single_cpu():
    _test_graph_single_device('cpu', 0)


def _keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(image_size, image_size), name='images'),
        tf.keras.layers.Flatten(input_shape=(image_size, image_size)),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(labels_size, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def _test_keras_single_device(device='cpu', device_id=0):
    skip_for_incompatible_tf()

    with tf.device('/{0}:{1}'.format(device, device_id)):
        model = _keras_model()

        train_dataset = _get_train_dataset(device, device_id)

        model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=iterations)

        assert model.evaluate(
            train_dataset,
            steps=iterations)[1] > target


def test_keras_single_gpu():
    _test_keras_single_device('gpu', 0)


def test_keras_single_other_gpu():
    _test_keras_single_device('gpu', 1)


def test_keras_single_cpu():
    _test_keras_single_device('cpu', 0)


def clear_checkpoints():
    remove_directory('/tmp/tensorflow-checkpoints', ignore_errors=True)


def _test_estimators_single_device(model, device='cpu', device_id=0):
    skip_for_incompatible_tf()

    with tf.device('/{0}:{1}'.format(device, device_id)):
        def train_fn():
            return _get_train_dataset(device, device_id).map(
                lambda features, labels: ({'images': features}, labels))

        model.train(input_fn=train_fn, steps=epochs * iterations)

        evaluation = model.evaluate(
            input_fn=train_fn,
            steps=iterations)
        final_accuracy = evaluation['acc'] if 'acc' in evaluation else evaluation['accuracy']
        print('Final accuracy: ', final_accuracy)

        assert final_accuracy > target


def _test_estimators_classifier_single_device(device='cpu', device_id=0):
    feature_columns = [tf.feature_column.numeric_column(
        "images", shape=[image_size, image_size])]

    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[hidden_size],
        n_classes=labels_size,
        dropout=dropout,
        model_dir='/tmp/tensorflow-checkpoints',
        optimizer=Adam())

    _test_estimators_single_device(model, device, device_id)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_gpu():

    _test_estimators_classifier_single_device('gpu', 0)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_other_gpu():
    _test_estimators_classifier_single_device('gpu', 1)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_cpu():
    _test_estimators_classifier_single_device('cpu', 0)


def _test_estimators_wrapping_keras_single_device(device='cpu', device_id=0):
    _test_estimators_single_device(
        tf.keras.estimator.model_to_estimator(keras_model=_keras_model()),
        device,
        device_id)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_wrapping_keras_single_gpu():
    _test_estimators_wrapping_keras_single_device('gpu', 0)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_wrapping_keras_single_other_gpu():
    _test_estimators_wrapping_keras_single_device('gpu', 1)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_wrapping_keras_single_cpu():
    _test_estimators_wrapping_keras_single_device('cpu', 0)
