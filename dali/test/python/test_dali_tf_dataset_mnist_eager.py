# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from test_dali_tf_dataset_mnist import *
from nose_utils import raises

tf.compat.v1.enable_eager_execution()

def test_keras_single_gpu():
    run_keras_single_device('gpu', 0)


def test_keras_single_other_gpu():
    run_keras_single_device('gpu', 1)


def test_keras_single_cpu():
    run_keras_single_device('cpu', 0)


@raises(Exception, "TF device and DALI device mismatch")
def test_keras_wrong_placement_gpu():
    with tf.device('cpu:0'):
        model = keras_model()
        train_dataset = get_dataset('gpu', 0)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=ITERATIONS)


@raises(Exception, "TF device and DALI device mismatch")
def test_keras_wrong_placement_cpu():
    with tf.device('gpu:0'):
        model = keras_model()
        train_dataset = get_dataset('cpu', 0)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=ITERATIONS)


@with_setup(skip_for_incompatible_tf)
def test_keras_multi_gpu_mirrored_strategy():
    strategy = tf.distribute.MirroredStrategy(devices=available_gpus())

    with strategy.scope():
        model = keras_model()

    train_dataset = get_dataset_multi_gpu(strategy)

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=ITERATIONS)

    assert model.evaluate(
        train_dataset,
        steps=ITERATIONS)[1] > TARGET


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_gpu():
    run_estimators_single_device('gpu', 0)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_other_gpu():
    run_estimators_single_device('gpu', 1)


@with_setup(clear_checkpoints, clear_checkpoints)
def test_estimators_single_cpu():
    run_estimators_single_device('cpu', 0)
