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
import tensorflow.compat.v1 as tf_v1
from nose_utils import with_setup, SkipTest, raises
import test_dali_tf_dataset_mnist as mnist
from packaging.version import Version

mnist.tf.compat.v1.disable_eager_execution()


@with_setup(tf.keras.backend.clear_session)
def test_keras_single_gpu():
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    mnist.run_keras_single_device("gpu", 0)


@with_setup(tf.keras.backend.clear_session)
def test_keras_single_other_gpu():
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    mnist.run_keras_single_device("gpu", 1)


@with_setup(tf.keras.backend.clear_session)
def test_keras_single_cpu():
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    mnist.run_keras_single_device("cpu", 0)


@raises(tf.errors.OpError, "TF device and DALI device mismatch. TF*: CPU, DALI*: GPU for output")
def test_keras_wrong_placement_gpu():
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    with tf.device("cpu:0"):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset("gpu", 0)

        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)


@raises(tf.errors.OpError, "TF device and DALI device mismatch. TF*: GPU, DALI*: CPU for output")
def test_keras_wrong_placement_cpu():
    if Version(tf.__version__) >= Version("2.16"):
        raise SkipTest("TF < 2.16 is required for this test")
    with tf.device("gpu:0"):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset("cpu", 0)

        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)


@with_setup(tf.compat.v1.reset_default_graph)
def test_graph_single_gpu():
    mnist.run_graph_single_device("gpu", 0)


@with_setup(tf.compat.v1.reset_default_graph)
def test_graph_single_cpu():
    mnist.run_graph_single_device("cpu", 0)


@with_setup(tf.compat.v1.reset_default_graph)
def test_graph_single_other_gpu():
    mnist.run_graph_single_device("gpu", 1)


# This function is copied form:
# github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L102
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf_v1.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf_v1.concat(grads, 0)
        grad = tf_v1.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


@with_setup(tf_v1.reset_default_graph)
def test_graph_multi_gpu():
    iterator_initializers = []

    with tf.device("/cpu:0"):
        tower_grads = []

        for i in range(mnist.num_available_gpus()):
            with tf.device("/gpu:{}".format(i)):
                daliset = mnist.get_dataset("gpu", i, i, mnist.num_available_gpus())

                iterator = tf_v1.data.make_initializable_iterator(daliset)
                iterator_initializers.append(iterator.initializer)
                images, labels = iterator.get_next()

                images = tf_v1.reshape(
                    images, [mnist.BATCH_SIZE, mnist.IMAGE_SIZE * mnist.IMAGE_SIZE]
                )
                labels = tf_v1.reshape(
                    tf_v1.one_hot(labels, mnist.NUM_CLASSES), [mnist.BATCH_SIZE, mnist.NUM_CLASSES]
                )

                logits_train = mnist.graph_model(images, reuse=(i != 0), is_training=True)
                logits_test = mnist.graph_model(images, reuse=True, is_training=False)

                loss_op = tf_v1.reduce_mean(
                    tf_v1.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=labels)
                )
                optimizer = tf_v1.train.AdamOptimizer()
                grads = optimizer.compute_gradients(loss_op)

                if i == 0:
                    correct_pred = tf_v1.equal(
                        tf_v1.argmax(logits_test, 1), tf_v1.argmax(labels, 1)
                    )
                    accuracy = tf_v1.reduce_mean(tf_v1.cast(correct_pred, tf_v1.float32))

                tower_grads.append(grads)

        tower_grads = average_gradients(tower_grads)
        train_step = optimizer.apply_gradients(tower_grads)

    mnist.train_graph(iterator_initializers, train_step, accuracy)


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_gpu():
    mnist.run_estimators_single_device("gpu", 0)


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_other_gpu():
    mnist.run_estimators_single_device("gpu", 1)


@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_cpu():
    mnist.run_estimators_single_device("cpu", 0)
