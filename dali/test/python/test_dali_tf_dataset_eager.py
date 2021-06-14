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

import tensorflow as tf
from nvidia.dali.plugin.tf.experimental import DALIDatasetWithInputs
import nvidia.dali.plugin.tf as dali_tf
from test_utils_tensorflow import *
from test_dali_tf_dataset_pipelines import *
from nose.tools import raises, with_setup
import random as random

tf.compat.v1.enable_eager_execution()


def test_tf_dataset_gpu():
    run_tf_dataset_eager_mode('gpu')


def test_tf_dataset_cpu():
    run_tf_dataset_eager_mode('cpu')


def run_tf_dataset_with_constant_input(dev, shape, value, dtype):
    tensor = np.full(shape, value, dtype)
    run_tf_dataset_eager_mode(dev,
        get_pipeline_desc=external_source_tester(shape, dtype, FixedSampleIterator(tensor)),
        to_dataset=external_source_converter_with_fixed_value(shape, dtype, tensor))

def test_tf_dataset_with_constant_input():
    for dev in ['cpu', 'gpu']:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                value = random.choice([42, 255])
                yield run_tf_dataset_with_constant_input, dev, shape, value, dtype


def run_tf_dataset_with_random_input(dev, max_shape, dtype):
    run_tf_dataset_eager_mode(
        dev,
        get_pipeline_desc=external_source_tester(max_shape, dtype,
                                                 RandomSampleIterator(max_shape, dtype(0))),
        to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype))


def test_tf_dataset_with_random_input():
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                yield run_tf_dataset_with_random_input, dev, max_shape, dtype


# Run with everything on GPU (External Source op as well)
def run_tf_dataset_with_random_input_gpu(max_shape, dtype):
    run_tf_dataset_eager_mode(
        "gpu",
        get_pipeline_desc=external_source_tester(max_shape, dtype,
                                                 RandomSampleIterator(max_shape, dtype(0)), "gpu"),
        to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype))


def test_tf_dataset_with_random_input_gpu():
    for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
        for dtype in [np.uint8, np.int32, np.float32]:
            yield run_tf_dataset_with_random_input_gpu, max_shape, dtype


def run_tf_dataset_with_stop_iter(dev, max_shape, dtype, stop_samples):
    run_tf_dataset_eager_mode(dev,
                              to_stop_iter=True,
                              get_pipeline_desc=external_source_tester(
                                  max_shape, dtype,
                                  RandomSampleIterator(max_shape,
                                                       dtype(0),
                                                       start=0,
                                                       stop=stop_samples)),
                              to_dataset=external_source_converter_with_callback(
                                  RandomSampleIterator, max_shape, dtype, 0, stop_samples))


def test_tf_dataset_with_stop_iter():
    batch_size = 12
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for iters in [1, 2, 3, 4, 5]:
                    yield run_tf_dataset_with_stop_iter, dev, max_shape, dtype, iters * batch_size - 3

def run_tf_dataset_multi_input(dev, start_values, input_names):
    run_tf_dataset_eager_mode(dev,
        get_pipeline_desc=external_source_tester_multiple(start_values, input_names),
        to_dataset=external_source_converter_multiple(start_values, input_names))


start_values = [[np.full((2, 4), 42, dtype=np.int64),
                 np.full((3, 5), 123.0, dtype=np.float32)],
                [np.full((3, 5), 3.14, dtype=np.float32)],
                [
                    np.full((2, 4), 42, dtype=np.int64),
                    np.full((3, 5), 666.0, dtype=np.float32),
                    np.full((1, 7), -5, dtype=np.int8)
                ]]

input_names = [["input_{}".format(i) for i, _ in enumerate(vals)] for vals in start_values]


def test_tf_dataset_multi_input():
    for dev in ['cpu', 'gpu']:
        for starts, names in zip(start_values, input_names):
            yield run_tf_dataset_multi_input, dev, starts, names


@raises(Exception)
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, 'cpu', 0)

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

    pipeline = get_image_pipeline(batch_size, num_threads, 'gpu', 0)

    with tf.device('/cpu:0'):
        dataset = get_dali_dataset_from_pipeline(
            pipeline, batch_size, num_threads, 'cpu', 0)

    for sample in dataset:
        pass

@raises(Exception)
def check_tf_dataset_mismatched_input_type(wrong_input_dataset, wrong_input_name, wrong_input_layout=None):
    pipe = many_input_pipeline(True, "cpu", None, ["a", "b"], batch_size=8, num_threads=4, device_id=0)

    with tf.device('/cpu:0'):
        input_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                input_datasets=wrong_input_dataset,
                input_names=wrong_input_name,
                input_layouts=wrong_input_layout,
                pipeline=pipe,
                batch_size=pipe.batch_size,
                output_shapes=(None, None),
                output_dtypes=(tf.int32, tf.int32),
                num_threads=pipe.num_threads,
                device_id=pipe.device_id)
        return dali_dataset

def test_tf_dataset_mismatched_input_type():
    input_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    for wrong_input_dataset in ["str", [input_dataset]]:
        yield check_tf_dataset_mismatched_input_type, wrong_input_dataset, "a"
    for wrong_input_name in [42, ["a"]]:
        yield check_tf_dataset_mismatched_input_type, input_dataset, wrong_input_name
    yield check_tf_dataset_mismatched_input_type, (input_dataset, input_dataset), ("a", "b"), "HWC"
    yield check_tf_dataset_mismatched_input_type, (input_dataset), ("a", "b")
    yield check_tf_dataset_mismatched_input_type, (input_dataset, input_dataset), ("b")



# Test if the TypeError is raised for unsupported arguments for regular DALIDataset
@raises(TypeError)
def test_tf_experimental_inputs_disabled():
    pipeline = get_image_pipeline(4, 4, 'cpu', 0)
    dali_tf.DALIDataset(pipeline,
                        input_datasets=tf.data.Dataset.from_tensors(np.int32([42, 42])),
                        input_names="test")


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
