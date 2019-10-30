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
    from tensorflow.compat.v1 import reset_default_graph
    from tensorflow.compat.v1 import variable_scope
    from tensorflow.compat.v1 import layers
    from tensorflow.compat.v1 import global_variables_initializer
    from tensorflow.compat.v1.data import make_initializable_iterator
except:
    # Older TF versions don't have compat.v1 layer
    from tensorflow import Session
    from tensorflow import placeholder
    from tensorflow import truncated_normal
    from tensorflow import reset_default_graph
    from tensorflow import variable_scope
    from tensorflow import layers
    from tensorflow import global_variables_initializer
    from tensorflow.data import make_initializable_iterator
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

target = 0.85
batch_size = 32
dropout = 0.2
image_size = 28
labels_size = 10
hidden_size = 128
epochs = 5
iterations = 100

data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')


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


def _get_mnist_dataset(device='cpu', device_id=0, shard_id=0, num_shards=1):
    mnist_pipeline = MnistPipeline(
        4, data_path, device, device_id, shard_id, num_shards)
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


def _get_train_dataset(device='cpu', device_id=0, shard_id=0, num_shards=1):
    return _get_mnist_dataset(
        device,
        device_id,
        shard_id,
        num_shards)


def _graph_model(images, reuse, is_training):
    with variable_scope('mnist_net', reuse=reuse):
        images = layers.flatten(images)
        images = layers.dense(images, hidden_size, activation=tf.nn.relu)
        images = layers.dropout(images, rate=dropout, training=is_training)
        images = layers.dense(images, labels_size, activation=tf.nn.softmax)

    return images


def _train_graph(iterator_initializers, train_op, accuracy):
    with Session() as sess:
        sess.run(global_variables_initializer())
        sess.run(iterator_initializers)

        for i in range(epochs * iterations):
            sess.run(train_op)
            if i % iterations == 0:
                train_accuracy = accuracy.eval()
                print("Step %d, accuracy: %g" % (i, train_accuracy))

        final_accuracy = 0
        for _ in range(iterations):
            final_accuracy = final_accuracy + \
                accuracy.eval()
        final_accuracy = final_accuracy / iterations

        print('Final accuracy: ', final_accuracy)
        assert final_accuracy > target


def _test_graph_single_device(device='cpu', device_id=0):
    skip_for_incompatible_tf()

    with tf.device('/{0}:{1}'.format(device, device_id)):
        daliset = _get_train_dataset(device, device_id)

        iterator = make_initializable_iterator(daliset)
        images, labels = iterator.get_next()

        images = tf.reshape(images, [batch_size, image_size*image_size])
        labels = tf.reshape(
            tf.one_hot(labels, labels_size),
            [batch_size, labels_size])

        logits_train = _graph_model(images, reuse=False, is_training=True)
        logits_test = _graph_model(images, reuse=True, is_training=False)

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_train, labels=labels))
        train_step = AdamOptimizer().minimize(loss_op)

        correct_pred = tf.equal(
            tf.argmax(logits_test, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    _train_graph([iterator.initializer], train_step, accuracy)


@with_setup(reset_default_graph)
def test_graph_single_gpu():
    _test_graph_single_device('gpu', 0)


@with_setup(reset_default_graph)
def test_graph_single_other_gpu():
    _test_graph_single_device('gpu', 1)


@with_setup(reset_default_graph)
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

    def train_fn():
        with tf.device('/{0}:{1}'.format(device, device_id)):
            return _get_train_dataset(device, device_id).map(
                lambda features, labels: ({'images': features}, labels))

    model.train(input_fn=train_fn, steps=epochs * iterations)

    def test_fn():
        return _get_train_dataset(device, device_id).map(
            lambda features, labels: ({'images': features}, labels))

    evaluation = model.evaluate(
        input_fn=test_fn,
        steps=iterations)
    final_accuracy = evaluation['acc'] if 'acc' in evaluation else evaluation['accuracy']
    print('Final accuracy: ', final_accuracy)

    assert final_accuracy > target


def _run_config(device='cpu', device_id=0):
    return tf.estimator.RunConfig(
        model_dir='/tmp/tensorflow-checkpoints',
        device_fn=lambda op: '/{0}:{1}'.format(device, device_id))


def _test_estimators_classifier_single_device(device='cpu', device_id=0):
    feature_columns = [tf.feature_column.numeric_column(
        "images", shape=[image_size, image_size])]

    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[hidden_size],
        n_classes=labels_size,
        dropout=dropout,
        config=_run_config(device, device_id),
        optimizer=Adam)

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
    with tf.device('/{0}:{1}'.format(device, device_id)):
        keras_model = _keras_model()
    model = tf.keras.estimator.model_to_estimator(
        keras_model=keras_model,
        config=_run_config(device, device_id))
    _test_estimators_single_device(
        model,
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


# This function is copied form: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L102
def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


@with_setup(reset_default_graph)
def test_graph_multi_gpu():
    iterator_initializers = []

    with tf.device('/cpu:0'):
        tower_grads = []

        for i in range(num_available_gpus()):
            with tf.device('/gpu:{}'.format(i)):
                daliset = _get_train_dataset(
                    'gpu', i, i, num_available_gpus())

                iterator = make_initializable_iterator(daliset)
                iterator_initializers.append(iterator.initializer)
                images, labels = iterator.get_next()

                images = tf.reshape(
                    images, [batch_size, image_size*image_size])
                labels = tf.reshape(
                    tf.one_hot(labels, labels_size),
                    [batch_size, labels_size])

                logits_train = _graph_model(
                    images, reuse=(i != 0), is_training=True)
                logits_test = _graph_model(
                    images, reuse=True, is_training=False)

                loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits_train, labels=labels))
                optimizer = AdamOptimizer()
                grads = optimizer.compute_gradients(loss_op)

                if i == 0:
                    correct_pred = tf.equal(
                        tf.argmax(logits_test, 1), tf.argmax(labels, 1))
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred, tf.float32))

                tower_grads.append(grads)

        tower_grads = _average_gradients(tower_grads)
        train_step = optimizer.apply_gradients(tower_grads)

    _train_graph(iterator_initializers, train_step, accuracy)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_multi_gpu():
    skip_for_incompatible_tf()

    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=available_gpus())

    def train_fn(input_context):
        return _get_train_dataset('cpu', 0).map(
            lambda features, labels: ({'images': features}, labels))

    config = tf.estimator.RunConfig(
        model_dir='/tmp/tensorflow-checkpoints',
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy)

    feature_columns = [tf.feature_column.numeric_column(
        "images", shape=[image_size, image_size])]

    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[hidden_size],
        n_classes=labels_size,
        dropout=dropout,
        optimizer=Adam,
        config=config)

    model.train(input_fn=train_fn, steps=epochs * iterations)

    evaluation = model.evaluate(
        input_fn=train_fn,
        steps=iterations)
    final_accuracy = evaluation['acc'] if 'acc' in evaluation else evaluation['accuracy']
    print('Final accuracy: ', final_accuracy)

    assert final_accuracy > target
