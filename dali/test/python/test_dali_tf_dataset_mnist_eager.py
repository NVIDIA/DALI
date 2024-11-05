# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorflow as tf
from nose_utils import with_setup, raises, SkipTest

import test_dali_tf_dataset_mnist as mnist
from test_utils_tensorflow import skip_for_incompatible_tf, available_gpus
from packaging.version import Version

tf.compat.v1.enable_eager_execution()


def test_keras_single_gpu():
    mnist.run_keras_single_device("gpu", 0)


def test_keras_single_other_gpu():
    mnist.run_keras_single_device("gpu", 1)


def test_keras_single_cpu():
    mnist.run_keras_single_device("cpu", 0)


@with_setup(skip_for_incompatible_tf)
@raises(tf.errors.OpError, "TF device and DALI device mismatch")
def test_keras_wrong_placement_gpu():
    with tf.device("cpu:0"):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset("gpu", 0)

        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)


@with_setup(skip_for_incompatible_tf)
@raises(tf.errors.OpError, "TF device and DALI device mismatch")
def test_keras_wrong_placement_cpu():
    with tf.device("gpu:0"):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset("cpu", 0)

        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)


@with_setup(skip_for_incompatible_tf)
def test_keras_multi_gpu_mirrored_strategy():
    # due to compatibility problems between the driver, cuda version and
    # TensorFlow 2.12 test_keras_multi_gpu_mirrored_strategy doesn't work.
    if Version(tf.__version__) >= Version("2.12.0"):
        raise SkipTest("This test is not supported for TensorFlow 2.12")
    strategy = tf.distribute.MirroredStrategy(devices=available_gpus())

    with strategy.scope():
        model = mnist.keras_model()

    train_dataset = mnist.get_dataset_multi_gpu(strategy)

    model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)

    assert model.evaluate(train_dataset, steps=mnist.ITERATIONS)[1] > mnist.TARGET


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_gpu():
    mnist.run_estimators_single_device("gpu", 0)


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_other_gpu():
    mnist.run_estimators_single_device("gpu", 1)


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_cpu():
    mnist.run_estimators_single_device("cpu", 0)
