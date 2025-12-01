# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.plugin.tf as dali_tf
import os
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from test_utils_tensorflow import num_available_gpus
from shutil import rmtree as remove_directory
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from packaging.version import Version
from nose_utils import SkipTest


TARGET = 0.8
BATCH_SIZE = 64
DROPOUT = 0.2
IMAGE_SIZE = 28
NUM_CLASSES = 10
HIDDEN_SIZE = 128
EPOCHS = 5
ITERATIONS = 100

data_path = os.path.join(os.environ["DALI_EXTRA_PATH"], "db/MNIST/training/")


def mnist_pipeline(num_threads, path, device, device_id=0, shard_id=0, num_shards=1, seed=0):
    pipeline = Pipeline(BATCH_SIZE, num_threads, device_id, seed)
    with pipeline:
        jpegs, labels = fn.readers.caffe2(
            path=path, random_shuffle=True, shard_id=shard_id, num_shards=num_shards
        )
        images = fn.decoders.image(
            jpegs, device="mixed" if device == "gpu" else "cpu", output_type=types.GRAY
        )
        if device == "gpu":
            labels = labels.gpu()
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=[0.0], std=[255.0], output_layout="CHW"
        )

        pipeline.set_outputs(images, labels)

    return pipeline


def get_dataset(device="cpu", device_id=0, shard_id=0, num_shards=1, fail_on_device_mismatch=True):
    pipeline = mnist_pipeline(4, data_path, device, device_id, shard_id, num_shards)
    shapes = ((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE), (BATCH_SIZE,))
    dtypes = (tf.float32, tf.int32)

    daliset = dali_tf.DALIDataset(
        pipeline=pipeline,
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        num_threads=4,
        device_id=device_id,
        fail_on_device_mismatch=fail_on_device_mismatch,
    )
    return daliset


def get_dataset_multi_gpu(strategy):
    def dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            return get_dataset("gpu", device_id, device_id, num_available_gpus())

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_fetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
    )

    train_dataset = strategy.distribute_datasets_from_function(dataset_fn, input_options)
    return train_dataset


def keras_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE), name="images"),
            tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE)),
            tf.keras.layers.Dense(HIDDEN_SIZE, activation="relu"),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def run_keras_single_device(device="cpu", device_id=0):
    with tf.device("/{0}:{1}".format(device, device_id)):
        model = keras_model()
        train_dataset = get_dataset(device, device_id)

        model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=ITERATIONS)

        assert model.evaluate(train_dataset, steps=ITERATIONS)[1] > TARGET


def graph_model(images, reuse, is_training):
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    with tf_v1.variable_scope("mnist_net", reuse=reuse):
        images = tf_v1.layers.flatten(images)
        images = tf_v1.layers.dense(images, HIDDEN_SIZE, activation=tf_v1.nn.relu)
        images = tf_v1.layers.dropout(images, rate=DROPOUT, training=is_training)
        images = tf_v1.layers.dense(images, NUM_CLASSES, activation=tf_v1.nn.softmax)
    return images


def train_graph(iterator_initializers, train_op, accuracy):
    with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        sess.run(iterator_initializers)

        for i in range(EPOCHS * ITERATIONS):
            sess.run(train_op)
            if i % ITERATIONS == 0:
                train_accuracy = accuracy.eval()
                print("Step %d, accuracy: %g" % (i, train_accuracy))

        final_accuracy = 0
        for _ in range(ITERATIONS):
            final_accuracy = final_accuracy + accuracy.eval()
        final_accuracy = final_accuracy / ITERATIONS

        print("Final accuracy: ", final_accuracy)
        assert final_accuracy > TARGET


def run_graph_single_device(device="cpu", device_id=0):
    with tf.device("/{0}:{1}".format(device, device_id)):
        daliset = get_dataset(device, device_id)

        iterator = tf_v1.data.make_initializable_iterator(daliset)
        images, labels = iterator.get_next()

        # images = tf_v1.reshape(images, [BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE])
        labels = tf_v1.reshape(tf_v1.one_hot(labels, NUM_CLASSES), [BATCH_SIZE, NUM_CLASSES])

        logits_train = graph_model(images, reuse=False, is_training=True)
        logits_test = graph_model(images, reuse=True, is_training=False)

        loss_op = tf_v1.reduce_mean(
            tf_v1.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=labels)
        )
        train_step = tf_v1.train.AdamOptimizer().minimize(loss_op)

        correct_pred = tf_v1.equal(tf_v1.argmax(logits_test, 1), tf_v1.argmax(labels, 1))
        accuracy = tf_v1.reduce_mean(tf_v1.cast(correct_pred, tf_v1.float32))

    train_graph([iterator.initializer], train_step, accuracy)


def clear_checkpoints():
    remove_directory("/tmp/tensorflow-checkpoints", ignore_errors=True)


def _test_estimators_single_device(model, device="cpu", device_id=0):
    def dataset_fn():
        with tf.device("/{0}:{1}".format(device, device_id)):
            return get_dataset(device, device_id)

    model.train(input_fn=dataset_fn, steps=EPOCHS * ITERATIONS)

    evaluation = model.evaluate(input_fn=dataset_fn, steps=ITERATIONS)
    final_accuracy = evaluation["acc"] if "acc" in evaluation else evaluation["accuracy"]
    print("Final accuracy: ", final_accuracy)

    assert final_accuracy > TARGET


def _run_config(device="cpu", device_id=0):
    return tf.estimator.RunConfig(
        model_dir="/tmp/tensorflow-checkpoints",
        device_fn=lambda op: "/{0}:{1}".format(device, device_id),
    )


def run_estimators_single_device(device="cpu", device_id=0):
    if Version(tf.__version__) < Version("2.16"):
        with tf.device("/{0}:{1}".format(device, device_id)):
            model = keras_model()
        model = tf.keras.estimator.model_to_estimator(
            keras_model=model, config=_run_config(device, device_id)
        )
        _test_estimators_single_device(model, device, device_id)
    else:
        raise SkipTest("TF < 2.16 is required for this test")
