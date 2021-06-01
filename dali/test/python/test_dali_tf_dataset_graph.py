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

import tensorflow
from test_utils_tensorflow import *
from test_dali_tf_dataset_pipelines import *
from nose.tools import raises, with_setup

tensorflow.compat.v1.disable_eager_execution()


def test_tf_dataset_gpu():
    run_tf_dataset_graph('gpu')


def test_tf_dataset_cpu():
    run_tf_dataset_graph('cpu')


def run_tf_dataset_with_fixed_input(dev, shape, value, dtype):
    tensor = np.full(shape, value, dtype)
    run_tf_dataset_graph(dev,
        get_pipeline_desc=external_source_tester(shape, dtype, FixedSampleIterator(tensor)),
        to_dataset=external_source_converter_with_fixed_value(shape, dtype, tensor))

def test_tf_dataset_with_fixed_input():
    for dev in ['cpu', 'gpu']:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for value in [42, 255]:
                    yield run_tf_dataset_with_fixed_input, dev, shape, value, dtype


def run_tf_dataset_with_random_input(dev, max_shape, dtype):
    run_tf_dataset_graph(dev,
        get_pipeline_desc=external_source_tester(max_shape, dtype, RandomSampleIterator(max_shape, dtype(0))),
        to_dataset=external_source_converter_with_callback(max_shape, dtype, RandomSampleIterator))

def test_tf_dataset_with_random_input():
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                yield run_tf_dataset_with_random_input, dev, max_shape, dtype


@raises(Exception)
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, 'cpu', 0)

    with tf.device('/gpu:0'):
        dataset = get_dali_dataset_from_pipeline(
            pipeline, batch_size, num_threads, 'gpu', 0)

    run_dataset_in_graph(dataset, iterations)


@raises(Exception)
def test_tf_dataset_wrong_placement_gpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, 'gpu', 0)

    with tf.device('/cpu:0'):
        dataset = get_dali_dataset_from_pipeline(
            pipeline, batch_size, num_threads, 'cpu', 0)

    run_dataset_in_graph(dataset, iterations)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    run_tf_dataset_graph('gpu', 1)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_multigpu_manual_placement():
    run_tf_dataset_multigpu_graph_manual_placement()
