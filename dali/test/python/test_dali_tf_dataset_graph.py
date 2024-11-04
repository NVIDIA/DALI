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

import itertools
import numpy as np
import random as random
import tensorflow as tf
from nose_utils import with_setup, raises
from test_dali_tf_dataset_pipelines import (
    FixedSampleIterator,
    external_source_tester,
    external_source_converter_with_fixed_value,
    external_source_converter_with_callback,
    RandomSampleIterator,
    external_source_converter_multiple,
    get_min_shape_helper,
    external_source_tester_multiple,
)
from test_dali_tf_es_pipelines import (
    external_source_to_tf_dataset,
    gen_tf_with_dali_external_source,
    get_external_source_pipe,
)
from test_utils_tensorflow import (
    run_tf_dataset_graph,
    skip_inputs_for_incompatible_tf,
    run_dataset_in_graph,
    run_tf_dataset_multigpu_graph_manual_placement,
    get_dali_dataset_from_pipeline,
    get_image_pipeline,
)

tf.compat.v1.disable_eager_execution()


def test_tf_dataset_gpu():
    run_tf_dataset_graph("gpu")


def test_tf_dataset_cpu():
    run_tf_dataset_graph("cpu")


def run_tf_dataset_with_constant_input(dev, shape, value, dtype, batch):
    tensor = np.full(shape, value, dtype)
    run_tf_dataset_graph(
        dev,
        get_pipeline_desc=external_source_tester(
            shape, dtype, FixedSampleIterator(tensor), batch=batch
        ),
        to_dataset=external_source_converter_with_fixed_value(shape, dtype, tensor, batch),
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_constant_input():
    for dev in ["cpu", "gpu"]:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ["dataset", True, False, None]:
                    value = random.choice([42, 255])
                    yield run_tf_dataset_with_constant_input, dev, shape, value, dtype, batch


def run_tf_dataset_with_random_input(dev, max_shape, dtype, batch):
    min_shape = get_min_shape_helper(batch, max_shape)
    iterator = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    run_tf_dataset_graph(
        dev,
        get_pipeline_desc=external_source_tester(max_shape, dtype, iterator, batch=batch),
        to_dataset=external_source_converter_with_callback(
            RandomSampleIterator, max_shape, dtype, 0, 1e10, min_shape, batch=batch
        ),
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input():
    for dev in ["cpu", "gpu"]:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ["dataset", True, False, None]:
                    yield run_tf_dataset_with_random_input, dev, max_shape, dtype, batch


# Run with everything on GPU (External Source op as well)
def run_tf_dataset_with_random_input_gpu(max_shape, dtype, batch):
    min_shape = get_min_shape_helper(batch, max_shape)
    iterator = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    run_tf_dataset_graph(
        "gpu",
        get_pipeline_desc=external_source_tester(max_shape, dtype, iterator, "gpu", batch=batch),
        to_dataset=external_source_converter_with_callback(
            RandomSampleIterator, max_shape, dtype, 0, 1e10, min_shape, batch=batch
        ),
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input_gpu():
    for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
        for dtype in [np.uint8, np.int32, np.float32]:
            for batch in ["dataset", True, False, None]:
                yield run_tf_dataset_with_random_input_gpu, max_shape, dtype, batch


def run_tf_dataset_no_copy(max_shape, dtype, dataset_dev, es_dev, no_copy):
    run_tf_dataset_graph(
        dataset_dev,
        get_pipeline_desc=external_source_tester(
            max_shape, dtype, RandomSampleIterator(max_shape, dtype(0)), es_dev, no_copy
        ),
        to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype),
    )


# Check if setting no_copy flags in all placement scenarios is ok as we override it internally
@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_no_copy():
    for max_shape in [(10, 20), (120, 120, 3)]:
        for dataset_dev in ["cpu", "gpu"]:
            for es_dev in ["cpu", "gpu"]:
                if dataset_dev == "cpu" and es_dev == "gpu":
                    continue  # GPU op in CPU dataset not supported
                for no_copy in [True, False, None]:
                    yield run_tf_dataset_no_copy, max_shape, np.uint8, dataset_dev, es_dev, no_copy


def run_tf_dataset_with_stop_iter(dev, max_shape, dtype, stop_samples):
    run_tf_dataset_graph(
        dev,
        to_stop_iter=True,
        get_pipeline_desc=external_source_tester(
            max_shape, dtype, RandomSampleIterator(max_shape, dtype(0), start=0, stop=stop_samples)
        ),
        to_dataset=external_source_converter_with_callback(
            RandomSampleIterator, max_shape, dtype, 0, stop_samples
        ),
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_stop_iter():
    batch_size = 12
    for dev in ["cpu", "gpu"]:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for iters in [1, 2, 3, 4, 5]:
                    yield (
                        run_tf_dataset_with_stop_iter,
                        dev,
                        max_shape,
                        dtype,
                        iters * batch_size - 3,
                    )


def run_tf_dataset_multi_input(dev, start_values, input_names, batches):
    run_tf_dataset_graph(
        dev,
        get_pipeline_desc=external_source_tester_multiple(start_values, input_names, batches),
        to_dataset=external_source_converter_multiple(start_values, input_names, batches),
    )


start_values = [
    [np.full((2, 4), -42, dtype=np.int64), np.full((3, 5), -123.0, dtype=np.float32)],
    [np.full((3, 5), -3.14, dtype=np.float32)],
    [
        np.full((2, 4), -42, dtype=np.int64),
        np.full((3, 5), -666.0, dtype=np.float32),
        np.full((1, 7), 5, dtype=np.int8),
    ],
]

input_names = [["input_{}".format(i) for i, _ in enumerate(vals)] for vals in start_values]


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_multi_input():
    for dev in ["cpu", "gpu"]:
        for starts, names in zip(start_values, input_names):
            yield run_tf_dataset_multi_input, dev, starts, names, ["dataset" for _ in input_names]
            for batches in list(itertools.product([True, False], repeat=len(input_names))):
                yield run_tf_dataset_multi_input, dev, starts, names, batches


def run_tf_with_dali_external_source(dev, es_args, ed_dev, dtype, *_):
    run_tf_dataset_graph(
        dev,
        get_pipeline_desc=get_external_source_pipe(es_args, dtype, ed_dev),
        to_dataset=external_source_to_tf_dataset,
        to_stop_iter=True,
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_with_dali_external_source():
    yield from gen_tf_with_dali_external_source(run_tf_with_dali_external_source)


tf_dataset_wrong_placement_error_msg = (
    r"TF device and DALI device mismatch. " r"TF device: [\w]*, DALI device: [\w]* for output"
)


@raises(Exception, regex=tf_dataset_wrong_placement_error_msg)
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, "cpu", 0)

    with tf.device("/gpu:0"):
        dataset = get_dali_dataset_from_pipeline(pipeline, "gpu", 0)

    run_dataset_in_graph(dataset, iterations)


@raises(Exception, regex=tf_dataset_wrong_placement_error_msg)
def test_tf_dataset_wrong_placement_gpu():
    batch_size = 12
    num_threads = 4
    iterations = 10

    pipeline = get_image_pipeline(batch_size, num_threads, "gpu", 0)

    with tf.device("/cpu:0"):
        dataset = get_dali_dataset_from_pipeline(pipeline, "cpu", 0)

    run_dataset_in_graph(dataset, iterations)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    run_tf_dataset_graph("gpu", 1)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_multigpu_manual_placement():
    run_tf_dataset_multigpu_graph_manual_placement()
