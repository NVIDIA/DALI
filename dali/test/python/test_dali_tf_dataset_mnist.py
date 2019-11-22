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
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except:
    pass


TARGET = 0.8
BATCH_SIZE = 64
DROPOUT = 0.2
IMAGE_SIZE = 28
NUM_CLASSES = 10
HIDDEN_SIZE = 128
EPOCHS = 5
ITERATIONS = 100

data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')


def setup():
    skip_for_incompatible_tf()


class MnistPipeline(Pipeline):
    def __init__(self, num_threads, path, device, device_id=0, shard_id=0, num_shards=1, seed=0):
        super(MnistPipeline, self).__init__(
            BATCH_SIZE, num_threads, device_id, seed)
        self.device = device
        self.reader = ops.Caffe2Reader(path=path, random_shuffle=True, shard_id=shard_id, num_shards=num_shards)
        self.decode = ops.ImageDecoder(
            device='mixed' if device is 'gpu' else 'cpu',
            output_type=types.GRAY)
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            output_dtype=types.FLOAT,
            image_type=types.GRAY,
            mean=[0.],
            std=[255.],
            output_layout="CHW")

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
        (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        (BATCH_SIZE)]
    dtypes = [
        tf.float32,
        tf.int32]

    daliset = dali_tf.DALIDataset(
        pipeline=mnist_pipeline,
        batch_size=BATCH_SIZE,
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
    with tf.variable_scope('mnist_net', reuse=reuse):
        images = tf.layers.flatten(images)
        images = tf.layers.dense(images, HIDDEN_SIZE, activation=tf.nn.relu)
        images = tf.layers.dropout(images, rate=DROPOUT, training=is_training)
        images = tf.layers.dense(images, NUM_CLASSES, activation=tf.nn.softmax)

    return images


def _train_graph(iterator_initializers, train_op, accuracy):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_initializers)

        for i in range(EPOCHS * ITERATIONS):
            sess.run(train_op)
            if i % ITERATIONS == 0:
                train_accuracy = accuracy.eval()
                print("Step %d, accuracy: %g" % (i, train_accuracy))

        final_accuracy = 0
        for _ in range(ITERATIONS):
            final_accuracy = final_accuracy + \
                accuracy.eval()
        final_accuracy = final_accuracy / ITERATIONS

        print('Final accuracy: ', final_accuracy)
        assert final_accuracy > TARGET


def _test_graph_single_device(device='cpu', device_id=0):
    with tf.device('/{0}:{1}'.format(device, device_id)):
        daliset = _get_train_dataset(device, device_id)

        iterator = tf.data.make_initializable_iterator(daliset)
        images, labels = iterator.get_next()

        images = tf.reshape(images, [BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE])
        labels = tf.reshape(
            tf.one_hot(labels, NUM_CLASSES),
            [BATCH_SIZE, NUM_CLASSES])

        logits_train = _graph_model(images, reuse=False, is_training=True)
        logits_test = _graph_model(images, reuse=True, is_training=False)

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_train, labels=labels))
        train_step = tf.train.AdamOptimizer().minimize(loss_op)

        correct_pred = tf.equal(
            tf.argmax(logits_test, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    _train_graph([iterator.initializer], train_step, accuracy)


@with_setup(tf.reset_default_graph)
def test_graph_single_gpu():
    _test_graph_single_device('gpu', 0)


@with_setup(tf.reset_default_graph)
def test_graph_single_other_gpu():
    _test_graph_single_device('gpu', 1)


@with_setup(tf.reset_default_graph)
def test_graph_single_cpu():
    _test_graph_single_device('cpu', 0)


def _keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE), name='images'),
        tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def _test_keras_single_device(device='cpu', device_id=0):
    with tf.device('/{0}:{1}'.format(device, device_id)):
        model = _keras_model()

        train_dataset = _get_train_dataset(device, device_id)

        model.fit(
            train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=ITERATIONS)

        assert model.evaluate(
            train_dataset,
            steps=ITERATIONS)[1] > TARGET


def test_keras_single_gpu():
    _test_keras_single_device('gpu', 0)


def test_keras_single_other_gpu():
    _test_keras_single_device('gpu', 1)


def test_keras_single_cpu():
    _test_keras_single_device('cpu', 0)


def clear_checkpoints():
    remove_directory('/tmp/tensorflow-checkpoints', ignore_errors=True)


def _test_estimators_single_device(model, device='cpu', device_id=0):
    def train_fn():
        with tf.device('/{0}:{1}'.format(device, device_id)):
            return _get_train_dataset(device, device_id).map(
                lambda features, labels: ({'images': features}, labels))

    model.train(input_fn=train_fn, steps=EPOCHS * ITERATIONS)

    def test_fn():
        return _get_train_dataset(device, device_id).map(
            lambda features, labels: ({'images': features}, labels))

    evaluation = model.evaluate(
        input_fn=test_fn,
        steps=ITERATIONS)
    final_accuracy = evaluation['acc'] if 'acc' in evaluation else evaluation['accuracy']
    print('Final accuracy: ', final_accuracy)

    assert final_accuracy > TARGET


def _run_config(device='cpu', device_id=0):
    return tf.estimator.RunConfig(
        model_dir='/tmp/tensorflow-checkpoints',
        device_fn=lambda op: '/{0}:{1}'.format(device, device_id))


def _test_estimators_classifier_single_device(device='cpu', device_id=0):
    feature_columns = [tf.feature_column.numeric_column(
        "images", shape=[IMAGE_SIZE, IMAGE_SIZE])]

    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[HIDDEN_SIZE],
        n_classes=NUM_CLASSES,
        dropout=DROPOUT,
        config=_run_config(device, device_id),
        optimizer=tf.train.AdamOptimizer)

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


@with_setup(tf.reset_default_graph)
def test_graph_multi_gpu():
    iterator_initializers = []

    with tf.device('/cpu:0'):
        tower_grads = []

        for i in range(num_available_gpus()):
            with tf.device('/gpu:{}'.format(i)):
                daliset = _get_train_dataset(
                    'gpu', i, i, num_available_gpus())

                iterator = tf.data.make_initializable_iterator(daliset)
                iterator_initializers.append(iterator.initializer)
                images, labels = iterator.get_next()

                images = tf.reshape(
                    images, [BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE])
                labels = tf.reshape(
                    tf.one_hot(labels, NUM_CLASSES),
                    [BATCH_SIZE, NUM_CLASSES])

                logits_train = _graph_model(
                    images, reuse=(i != 0), is_training=True)
                logits_test = _graph_model(
                    images, reuse=True, is_training=False)

                loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits_train, labels=labels))
                optimizer = tf.train.AdamOptimizer()
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


# Note: This picks up single Dataset instance on the CPU and distributes the data automatically.
# TODO(awolant): Goal is to figure out how to have GPU instance per replica on one machine.
def test_keras_multi_gpu():
    train_dataset = _get_train_dataset('cpu', 0).unbatch().batch(BATCH_SIZE * num_available_gpus())
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=available_gpus())

    with mirrored_strategy.scope():
        model = _keras_model()

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=ITERATIONS)

    assert model.evaluate(
        train_dataset,
        steps=ITERATIONS)[1] > TARGET


def _test_estimators_multi_gpu(model):
    def train_fn(input_context):
        return _get_train_dataset('cpu', 0).map(
            lambda features, labels: ({'images': features}, labels))

    model.train(input_fn=train_fn, steps=EPOCHS * ITERATIONS)

    evaluation = model.evaluate(
        input_fn=train_fn,
        steps=ITERATIONS)
    final_accuracy = evaluation['acc'] if 'acc' in evaluation else evaluation['accuracy']
    print('Final accuracy: ', final_accuracy)

    assert final_accuracy > TARGET


def _multi_gpu_classifier():
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=available_gpus())

    config = tf.estimator.RunConfig(
        model_dir='/tmp/tensorflow-checkpoints',
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy)

    feature_columns = [tf.feature_column.numeric_column(
        "images", shape=[IMAGE_SIZE, IMAGE_SIZE])]

    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[HIDDEN_SIZE],
        n_classes=NUM_CLASSES,
        dropout=DROPOUT,
        optimizer=tf.train.AdamOptimizer,
        config=config)
    return model


def _multi_gpu_keras_classifier():
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=available_gpus())

    config = tf.estimator.RunConfig(
        model_dir='/tmp/tensorflow-checkpoints',
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy)

    with mirrored_strategy.scope():
        keras_model = _keras_model()
    model = tf.keras.estimator.model_to_estimator(
        keras_model=keras_model,
        config=config)
    return model


# Note: This picks up single Dataset instance on the CPU and distributes the data automatically.
# TODO(awolant): Goal is to figure out how to have GPU instance per replica on one machine.
@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_multi_gpu():
    model = _multi_gpu_classifier()
    _test_estimators_multi_gpu(model)


# Note: This picks up single Dataset instance on the CPU and distributes the data automatically.
# TODO(awolant): Goal is to figure out how to have GPU instance per replica on one machine.
@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_wrapping_keras_multi_gpu():
    model = _multi_gpu_keras_classifier()
    _test_estimators_multi_gpu(model)
