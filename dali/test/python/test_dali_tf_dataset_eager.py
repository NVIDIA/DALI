# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.plugin.tf as dali_tf
from test_utils_tensorflow import *
from nose.tools import raises, with_setup

tf.compat.v1.enable_eager_execution()


def test_tf_dataset_gpu():
    run_tf_dataset_eager_mode('gpu')


def test_tf_dataset_cpu():
    run_tf_dataset_eager_mode('cpu')


@raises(Exception)
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_pipeline(batch_size, num_threads, 'cpu', 0)

    with tf.device('/gpu:0'):
        dataset = get_dali_dataset_from_pipeline(
            pipeline, batch_size, num_threads, 'gpu', 0)

    for sample in dataset:
        pass


@raises(Exception)
def test_tf_dataset_wrong_placement_gpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_pipeline(batch_size, num_threads, 'gpu', 0)

    with tf.device('/cpu:0'):
        dataset = get_dali_dataset_from_pipeline(
            pipeline, batch_size, num_threads, 'cpu', 0)

    for sample in dataset:
        pass


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    run_tf_dataset_eager_mode('gpu', 1)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_multigpu_manual_placement():
    run_tf_dataset_multigpu_eager_manual_placement()


# This test should be private (name starts with _) as it is called separately in L1
@with_setup(skip_for_incompatible_tf)
def _test_tf_dataset_multigpu_mirrored_strategy():
    run_tf_dataset_multigpu_eager_mirrored_strategy()
