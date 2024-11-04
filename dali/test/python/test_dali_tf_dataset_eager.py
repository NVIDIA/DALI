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
import numpy as np
from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.plugin.tf.experimental import Input
from nvidia.dali import fn
from nose_utils import with_setup, raises
from test_dali_tf_dataset_pipelines import (
    FixedSampleIterator,
    RandomSampleIterator,
    external_source_converter_multiple,
    external_source_converter_with_callback,
    external_source_converter_with_fixed_value,
    external_source_tester,
    external_source_tester_multiple,
    get_min_shape_helper,
    many_input_pipeline,
)
from test_dali_tf_es_pipelines import (
    external_source_to_tf_dataset,
    gen_tf_with_dali_external_source,
    get_external_source_pipe,
)
from test_utils_tensorflow import (
    get_dali_dataset_from_pipeline,
    get_image_pipeline,
    get_mix_size_image_pipeline,
    run_dataset_eager_mode,
    run_tf_dataset_eager_mode,
    run_tf_dataset_multigpu_eager_manual_placement,
    run_tf_dataset_multigpu_eager_mirrored_strategy,
    skip_for_incompatible_tf,
    skip_inputs_for_incompatible_tf,
)
import random as random
import itertools


tf.compat.v1.enable_eager_execution()


def test_tf_dataset_gpu():
    run_tf_dataset_eager_mode("gpu")


def test_tf_dataset_cpu():
    run_tf_dataset_eager_mode("cpu")


# Return differently sized images to check if DALIDataset can handle this case gracefully


@raises(
    tf.errors.FailedPreconditionError,
    glob="Batch output at index * from DALI pipeline is not uniform",
)
def test_mixed_size_pipeline():
    run_tf_dataset_eager_mode("gpu", get_pipeline_desc=get_mix_size_image_pipeline)


def run_tf_dataset_with_constant_input(dev, shape, value, dtype, batch):
    tensor = np.full(shape, value, dtype)
    get_pipeline_desc = external_source_tester(
        shape, dtype, FixedSampleIterator(tensor), batch=batch
    )
    to_dataset = external_source_converter_with_fixed_value(shape, dtype, tensor, batch)
    run_tf_dataset_eager_mode(dev, get_pipeline_desc=get_pipeline_desc, to_dataset=to_dataset)


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_constant_input():
    for dev in ["cpu", "gpu"]:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ["dataset", True, False, None]:
                    value = random.choice([42, 255])
                    yield run_tf_dataset_with_constant_input, dev, shape, value, dtype, batch


def run_tf_dataset_with_random_input(dev, max_shape, dtype, batch="dataset"):
    min_shape = get_min_shape_helper(batch, max_shape)
    it = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    get_pipeline_desc = external_source_tester(max_shape, dtype, it, batch=batch)
    to_dataset = external_source_converter_with_callback(
        RandomSampleIterator, max_shape, dtype, 0, 1e10, min_shape, batch=batch
    )
    run_tf_dataset_eager_mode(dev, get_pipeline_desc=get_pipeline_desc, to_dataset=to_dataset)


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input():
    for dev in ["cpu", "gpu"]:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ["dataset", False, True, None]:
                    yield run_tf_dataset_with_random_input, dev, max_shape, dtype, batch


# Run with everything on GPU (External Source op as well)
def run_tf_dataset_with_random_input_gpu(max_shape, dtype, batch):
    min_shape = get_min_shape_helper(batch, max_shape)
    it = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    get_pipeline_desc = external_source_tester(max_shape, dtype, it, "gpu", batch=batch)
    to_dataset = external_source_converter_with_callback(
        RandomSampleIterator, max_shape, dtype, 0, 1e10, min_shape, batch=batch
    )
    run_tf_dataset_eager_mode("gpu", get_pipeline_desc=get_pipeline_desc, to_dataset=to_dataset)


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input_gpu():
    for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
        for dtype in [np.uint8, np.int32, np.float32]:
            for batch in ["dataset", False, True, None]:
                yield run_tf_dataset_with_random_input_gpu, max_shape, dtype, batch


def run_tf_dataset_no_copy(max_shape, dtype, dataset_dev, es_dev, no_copy):
    get_pipeline_desc = external_source_tester(
        max_shape, dtype, RandomSampleIterator(max_shape, dtype(0)), es_dev, no_copy
    )
    to_dataset = external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype)
    run_tf_dataset_eager_mode(
        dataset_dev, get_pipeline_desc=get_pipeline_desc, to_dataset=to_dataset
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
    it1 = RandomSampleIterator(max_shape, dtype(0), start=0, stop=stop_samples)
    get_pipeline_desc = external_source_tester(max_shape, dtype, it1)
    to_dataset = external_source_converter_with_callback(
        RandomSampleIterator, max_shape, dtype, 0, stop_samples
    )
    run_tf_dataset_eager_mode(
        dev, to_stop_iter=True, get_pipeline_desc=get_pipeline_desc, to_dataset=to_dataset
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
    run_tf_dataset_eager_mode(
        dev,
        get_pipeline_desc=external_source_tester_multiple(start_values, input_names, batches),
        to_dataset=external_source_converter_multiple(start_values, input_names, batches),
    )


start_values = [
    [np.full((2, 4), 42, dtype=np.int64), np.full((3, 5), 123.0, dtype=np.float32)],
    [np.full((3, 5), 3.14, dtype=np.float32)],
    [
        np.full((2, 4), 42, dtype=np.int64),
        np.full((3, 5), 666.0, dtype=np.float32),
        np.full((1, 7), -5, dtype=np.int8),
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


@raises(tf.errors.InternalError, glob="TF device and DALI device mismatch")
def test_tf_dataset_wrong_placement_cpu():
    batch_size = 12
    num_threads = 4

    pipeline = get_image_pipeline(batch_size, num_threads, "cpu", 0)

    with tf.device("/gpu:0"):
        dataset = get_dali_dataset_from_pipeline(pipeline, "gpu", 0)

    for sample in dataset:
        pass


@raises(tf.errors.InternalError, glob="TF device and DALI device mismatch")
def test_tf_dataset_wrong_placement_gpu():
    batch_size = 12
    num_threads = 4

    pipeline = get_image_pipeline(batch_size, num_threads, "gpu", 0)

    with tf.device("/cpu:0"):
        dataset = get_dali_dataset_from_pipeline(pipeline, "cpu", 0)

    for sample in dataset:
        pass


def check_basic_dataset_build(input_datasets):
    input_names = ["a", "b"]
    batches = ["dataset" for _ in input_names]
    pipe = many_input_pipeline(
        True, "cpu", None, input_names, batches, batch_size=8, num_threads=4, device_id=0
    )

    with tf.device("/cpu:0"):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            input_datasets=input_datasets,
            pipeline=pipe,
            batch_size=pipe.max_batch_size,
            output_shapes=(None, None),
            output_dtypes=(tf.int32, tf.int32),
            num_threads=pipe.num_threads,
            device_id=pipe.device_id,
        )
        return dali_dataset


@raises(
    TypeError,
    glob="`input_datasets` must be a dictionary that maps input names * to input datasets",
)
def check_tf_dataset_wrong_input_type(wrong_input_datasets):
    check_basic_dataset_build(wrong_input_datasets)


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_wrong_input_type():
    input_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    # wrong `input_datasets` type (no dictionary)
    for wrong_input_dataset in ["a", input_dataset, [input_dataset]]:
        yield check_tf_dataset_wrong_input_type, wrong_input_dataset
    # wrong values in dictionary
    for wrong_input_dataset in ["str", [input_dataset]]:
        yield check_tf_dataset_wrong_input_type, {
            "a": wrong_input_dataset,
            "b": wrong_input_dataset,
        }
    # wrong keys in dictionary
    for wrong_input_name in [42, ("a", "b")]:
        yield check_tf_dataset_wrong_input_type, {wrong_input_name: input_dataset}


@raises(
    ValueError,
    glob="Found External Source nodes in the Pipeline, that were not assigned any inputs.",
)
@with_setup(skip_for_incompatible_tf)
def test_input_not_provided():
    input_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    check_basic_dataset_build({"a": input_dataset})


@raises(
    ValueError, glob="Did not find an External Source placeholder node * in the provided pipeline"
)
@with_setup(skip_for_incompatible_tf)
def test_missing_es_node():
    input_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    check_basic_dataset_build({"a": input_dataset, "b": input_dataset, "c": input_dataset})


@pipeline_def(batch_size=10, num_threads=4, device_id=0)
def es_pipe(kwargs):
    return fn.external_source(**kwargs)


def check_single_es_pipeline(kwargs, input_datasets):
    pipe = es_pipe(kwargs)

    with tf.device("/cpu:0"):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            input_datasets=input_datasets,
            pipeline=pipe,
            batch_size=pipe.max_batch_size,
            output_shapes=(None, None),
            output_dtypes=(tf.int32, tf.int32),
            num_threads=pipe.num_threads,
            device_id=pipe.device_id,
        )
        return dali_dataset


@raises(
    ValueError, glob="Did not find an External Source placeholder node * in the provided pipeline"
)
@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_es_with_source():
    in_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    check_single_es_pipeline({"name": "a", "source": []}, {"a": in_dataset})


@raises(
    ValueError,
    glob="The parameter ``num_outputs`` is only valid when using ``source`` to provide data.",
)
@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_es_num_outputs_provided():
    in_dataset = tf.data.Dataset.from_tensors(np.full((2, 2), 42)).repeat()
    check_single_es_pipeline({"name": "a", "num_outputs": 1}, {"a": in_dataset})


@raises(
    ValueError, glob="Found placeholder External Source node * in the Pipeline that was not named"
)
@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_disallowed_es():
    check_single_es_pipeline({}, {})


def check_layout(kwargs, input_datasets, layout):
    pipe = Pipeline(10, 4, 0)
    with pipe:
        input = fn.external_source(**kwargs)
        # Rely on the Pad internal check to ensure that External Source set layout
        pipe.set_outputs(fn.pad(input, axis_names=layout))

    with tf.device("/cpu:0"):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            input_datasets=input_datasets,
            pipeline=pipe,
            batch_size=pipe.max_batch_size,
            output_shapes=None,
            output_dtypes=tf.int64,
            num_threads=pipe.num_threads,
            device_id=pipe.device_id,
        )

    run_dataset_eager_mode(dali_dataset, 10)


def run_tf_with_dali_external_source(dev, es_args, ed_dev, dtype, *_):
    run_tf_dataset_eager_mode(
        dev,
        get_pipeline_desc=get_external_source_pipe(es_args, dtype, ed_dev),
        to_dataset=external_source_to_tf_dataset,
        to_stop_iter=True,
    )


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_with_dali_external_source():
    yield from gen_tf_with_dali_external_source(run_tf_with_dali_external_source)


@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_layouts():
    for shape, layout in [((2, 3), "XY"), ((10, 20, 3), "HWC"), ((4, 128, 64, 3), "FHWC")]:
        in_dataset = tf.data.Dataset.from_tensors(np.full(shape, 42)).repeat()
        # Captured from pipeline
        yield check_layout, {"layout": layout, "name": "in"}, {"in": in_dataset}, layout
        # Captured from pipeline
        yield check_layout, {"layout": layout, "name": "in"}, {"in": Input(in_dataset)}, layout
        # Set via experimental.Input, not specified in external source
        yield check_layout, {"name": "in"}, {"in": Input(in_dataset, layout=layout)}, layout


# Test if the TypeError is raised for unsupported arguments for regular DALIDataset
@raises(TypeError, glob="Dataset inputs are allowed only in *DALIDatasetWithInputs")
def test_tf_experimental_inputs_disabled():
    pipeline = get_image_pipeline(4, 4, "cpu", 0)
    dali_tf.DALIDataset(
        pipeline, input_datasets={"test": tf.data.Dataset.from_tensors(np.int32([42, 42]))}
    )


# Test if the ValueError is raised for external source with `source`.
@raises(
    ValueError, glob="DALIDataset got a DALI pipeline containing External Source operator nodes"
)
def test_tf_experimental_source_disabled():
    pipe = Pipeline(10, 4, 0)
    with pipe:
        input = fn.external_source(source=lambda: np.full((4, 4), 0), batch=False)
        pipe.set_outputs(fn.pad(input))
    dali_tf.DALIDataset(pipe, output_dtypes=tf.int32)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_other_gpu():
    run_tf_dataset_eager_mode("gpu", 1)


# This test should be private (name starts with _) as it is called separately in L1
def _test_tf_dataset_multigpu_manual_placement():
    run_tf_dataset_multigpu_eager_manual_placement()


# This test should be private (name starts with _) as it is called separately in L1
@with_setup(skip_for_incompatible_tf)
def _test_tf_dataset_multigpu_mirrored_strategy():
    run_tf_dataset_multigpu_eager_mirrored_strategy()
