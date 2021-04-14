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

import logging
# logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow').setLevel(logging.WARNING)

import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
from test_utils_tensorflow import *
from test_dali_tf_dataset_pipelines import *
from nose.tools import raises, with_setup

import numpy as np


tf.compat.v1.enable_eager_execution()

# def test_tf_dataset_cpu_to_gpu():
#     run_tf_dataset_eager_mode('gpu',
#         get_pipeline_desc=external_source_tester((10, 20), np.uint8),
#         to_dataset=external_source_cpu_to_gpu((10, 20), np.uint8))


# def test_tf_dataset_gpu_custom():
#     run_tf_dataset_eager_mode('gpu',
#         get_pipeline_desc=external_source_tester((10, 20), np.uint8),
#         to_dataset=external_source_converter_gpu((10, 20), np.uint8))

def run_tf_dataset_with_fixed_input(dev, shape, value, dtype):
    tensor = np.full(shape, value, dtype)
    run_tf_dataset_eager_mode(dev,
        get_pipeline_desc=external_source_tester(shape, dtype, FixedSampleIterator(tensor + 1)),
        to_dataset=external_source_converter_with_fixed_value(shape, dtype, tensor, '/cpu:0'))

def test_tf_dataset_with_fixed_input():
    for dev in ['cpu', 'gpu']:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for value in [42, 255]:
                    yield run_tf_dataset_with_fixed_input, dev, shape, value, dtype

def test_tf_dataset_tensor():
    for dev in ['gpu']:
        for shape in [(7, 42)]:
            for dtype in [np.uint8]:
                for value in [42]:
                    yield run_tf_dataset_with_fixed_input, dev, shape, value, dtype


def run_tf_dataset_with_random_input(dev, max_shape, dtype):
    run_tf_dataset_eager_mode(dev,
        get_pipeline_desc=external_source_tester(max_shape, dtype, RandomSampleIterator(max_shape, dtype(0))),
        to_dataset=external_source_converter_with_callback(max_shape, dtype, RandomSampleIterator, '/cpu:0'))

def test_tf_dataset_with_random_input():
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                yield run_tf_dataset_with_random_input, dev, max_shape, dtype

def test_tf_dataset_gpu_generator():
    for dev in ['gpu']:
        for max_shape in [(10, 20)]:
            for dtype in [np.uint8]:
                yield run_tf_dataset_with_random_input, dev, max_shape, dtype


def test_tf_dataset_gpu():
    run_tf_dataset_eager_mode('gpu')


def test_tf_dataset_cpu():
    run_tf_dataset_eager_mode('cpu')

def test_tf_dataset_cpu_in():
    run_tf_dataset_eager_mode_in('cpu')


@raises(Exception)
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, 'cpu', 0)

    with tf.device('/gpu:0'):
        dataset = to_image_dataset(pipeline)

    for sample in dataset:
        pass


@raises(Exception)
def test_tf_dataset_wrong_placement_gpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, 'gpu', 0)

    with tf.device('/cpu:0'):
        dataset = to_image_dataset(pipeline)

    for sample in dataset:
        pass


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    run_tf_dataset_eager_mode('gpu', 1)


# This test should be private (name starts with _) as it is called separately in L1
def test_tf_dataset_multigpu_manual_placement():
    run_tf_dataset_multigpu_eager_manual_placement()


# This test should be private (name starts with _) as it is called separately in L1
@with_setup(skip_for_incompatible_tf)
def test_tf_dataset_multigpu_mirrored_strategy():
    run_tf_dataset_multigpu_eager_mirrored_strategy()
