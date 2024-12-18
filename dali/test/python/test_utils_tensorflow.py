# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import os
import tensorflow as tf
from contextlib import contextmanager
from nose_utils import SkipTest
from nvidia.dali.pipeline import Pipeline
from tensorflow.python.client import device_lib

from test_utils import to_array, get_dali_extra_path


def skip_for_incompatible_tf():
    if not dali_tf.dataset_distributed_compatible_tensorflow():
        raise SkipTest("This feature is enabled for TF 2.5.0 and higher")


def skip_inputs_for_incompatible_tf():
    if not dali_tf.dataset_inputs_compatible_tensorflow():
        raise SkipTest("This feature is enabled for TF 2.4.1 and higher")


def num_available_gpus():
    local_devices = device_lib.list_local_devices()
    num_gpus = sum(1 for device in local_devices if device.device_type == "GPU")
    if not math.log2(num_gpus).is_integer():
        raise RuntimeError("Unsupported number of GPUs. This test can run on: 2^n GPUs.")
    return num_gpus


def available_gpus():
    devices = []
    for device_id in range(num_available_gpus()):
        devices.append("/gpu:{0}".format(device_id))
    return devices


@contextmanager
def expect_iter_end(should_raise, exception_type):
    try:
        yield
    except exception_type:
        if should_raise:
            raise


# ################################################################################################ #
#
# To test custom DALI pipeline and DALIDataset wrapper for it all the `run_tf_dataset_*`
# routines accept two arguments:
#   * get_pipeline_desc
#   * to_dataset
# Both are callbacks, examples of those are respectively:
#   * get_image_pipeline
#   * to_image_dataset
#
# Respective signatures:
# get_pipeline_desc(batch_size, num_threads, device, device_id, shard_id, num_shards,
#                  def_for_dataset) -> nvidia.dali.pipeline, shapes, dtypes
# `def_for_dataset` - indicates if this function is invoked to create a baseline standalone
#                     pipeline (False), or it will be wrapped into TF dataset (True)
# It is supposed to return a tuple that also describes the shapes and dtypes returned by the pipe.
#
#
# to_image_dataset(image_pipeline_desc, device_str) -> tf.data.Dataset
# image_pipeline_desc will be the tuple returned by the `get_pipeline_desc`, device_str
# is the expected placement of the tested DALIDataset
#
# ################################################################################################ #


def get_mix_size_image_pipeline(
    batch_size, num_threads, device, device_id=0, shard_id=0, num_shards=1, def_for_dataset=False
):
    test_data_root = get_dali_extra_path()
    file_root = os.path.join(test_data_root, "db", "coco_dummy", "images")
    annotations_file = os.path.join(test_data_root, "db", "coco_dummy", "instances.json")

    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, _, _, image_ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=False,
            image_ids=True,
        )
        images = fn.decoders.image(
            jpegs, device=("mixed" if device == "gpu" else "cpu"), output_type=types.RGB
        )

        pipe.set_outputs(images)

    shapes = ((batch_size, None, None, None),)
    dtypes = (tf.float32,)

    return pipe, shapes, dtypes


def get_image_pipeline(
    batch_size, num_threads, device, device_id=0, shard_id=0, num_shards=1, def_for_dataset=False
):
    test_data_root = get_dali_extra_path()
    file_root = os.path.join(test_data_root, "db", "coco_dummy", "images")
    annotations_file = os.path.join(test_data_root, "db", "coco_dummy", "instances.json")

    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, _, _, image_ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=False,
            image_ids=True,
        )
        images = fn.decoders.image(
            jpegs, device=("mixed" if device == "gpu" else "cpu"), output_type=types.RGB
        )
        images = fn.resize(images, resize_x=224, resize_y=224, interp_type=types.INTERP_LINEAR)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=[128.0, 128.0, 128.0], std=[1.0, 1.0, 1.0]
        )
        if device == "gpu":
            image_ids = image_ids.gpu()
        ids_reshaped = fn.reshape(image_ids, shape=[1, 1])
        ids_int16 = fn.cast(image_ids, dtype=types.INT16)

        pipe.set_outputs(images, ids_reshaped, ids_int16)

    shapes = ((batch_size, 3, 224, 224), (batch_size, 1, 1), (batch_size, 1))
    dtypes = (tf.float32, tf.int32, tf.int16)

    return pipe, shapes, dtypes


def to_image_dataset(image_pipeline_desc, device_str):
    dataset_pipeline, shapes, dtypes = image_pipeline_desc
    with tf.device(device_str):
        dali_dataset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=dataset_pipeline.batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            num_threads=dataset_pipeline.num_threads,
            device_id=dataset_pipeline.device_id,
        )
    return dali_dataset


def get_dali_dataset_from_pipeline(pipeline_desc, device, device_id, to_dataset=to_image_dataset):
    dali_dataset = to_dataset(pipeline_desc, "/{0}:{1}".format(device, device_id))
    return dali_dataset


def get_dali_dataset(
    batch_size,
    num_threads,
    device,
    device_id,
    num_devices=1,
    get_pipeline_desc=get_image_pipeline,
    to_dataset=to_image_dataset,
):
    shard_id = 0 if num_devices == 1 else device_id
    dataset_pipeline = get_pipeline_desc(
        batch_size, num_threads, device, device_id, shard_id, num_devices, def_for_dataset=True
    )

    return get_dali_dataset_from_pipeline(dataset_pipeline, device, device_id, to_dataset)


def get_pipe_dataset(
    batch_size,
    num_threads,
    device,
    device_id,
    num_devices=1,
    *,
    dali_on_dev_0=True,
    get_pipeline_desc=get_image_pipeline,
    to_dataset=to_image_dataset,
):
    shard_id = 0 if num_devices == 1 else device_id

    tf_dataset = get_dali_dataset(
        batch_size,
        num_threads,
        device,
        device_id,
        num_devices=num_devices,
        get_pipeline_desc=get_pipeline_desc,
        to_dataset=to_dataset,
    )

    dali_pipeline, _, _ = get_pipeline_desc(
        batch_size,
        num_threads,
        device,
        0 if dali_on_dev_0 else device_id,
        shard_id,
        num_devices,
        def_for_dataset=False,
    )

    return dali_pipeline, tf_dataset


def run_dataset_in_graph(dali_datasets, iterations, to_stop_iter=False):
    if not isinstance(dali_datasets, list):
        dali_datasets = [dali_datasets]

    dataset_results = []
    initializers = [tf.compat.v1.global_variables_initializer()]
    ops_to_run = []

    for dali_dataset in dali_datasets:
        iterator = tf.compat.v1.data.make_initializable_iterator(dali_dataset)
        initializers.append(iterator.initializer)
        ops_to_run.append(iterator.get_next())

    with tf.compat.v1.Session() as sess:
        sess.run(initializers)
        with expect_iter_end(not to_stop_iter, tf.errors.OutOfRangeError):
            for _ in range(iterations):
                dataset_results.append(sess.run(ops_to_run))
    return dataset_results


def run_dataset_eager_mode(dali_datasets, iterations, to_stop_iter=False):
    if not isinstance(dali_datasets, list):
        dali_datasets = [dali_datasets]

    results = []
    with expect_iter_end(not to_stop_iter, StopIteration):
        for i, batch in zip(range(iterations), zip(*dali_datasets)):
            results.append(batch)
    return results


def run_pipeline(pipelines, iterations, device, to_stop_iter=False):
    if not isinstance(pipelines, list):
        pipelines = [pipelines]
    results = []
    with expect_iter_end(not to_stop_iter, StopIteration):
        for _ in range(iterations):
            shard_outputs = []
            for pipeline in pipelines:
                pipe_outputs = pipeline.run()
                shard_outputs.append(tuple(to_array(result) for result in pipe_outputs))
            results.append(tuple(shard_outputs))
    return results


def compare(dataset_results, standalone_results, iterations=-1, num_devices=1):
    if iterations == -1:
        iterations = len(standalone_results)

    # list [iterations] of tuple [devices] of tuple [outputs] of tensors representing batch
    assert (
        len(dataset_results) == iterations
    ), f"Got {len(dataset_results)} dataset results for {iterations} iterations"
    for it in range(iterations):
        for device_id in range(num_devices):
            for tf_data, dali_data in zip(
                dataset_results[it][device_id], standalone_results[it][device_id]
            ):
                np.testing.assert_array_equal(
                    tf_data, dali_data, f"Iteration {it}, x = tf_data, y = DALI baseline"
                )


def run_tf_dataset_graph(
    device,
    device_id=0,
    get_pipeline_desc=get_image_pipeline,
    to_dataset=to_image_dataset,
    to_stop_iter=False,
):
    tf.compat.v1.reset_default_graph()
    batch_size = 12
    num_threads = 4
    iterations = 10

    standalone_pipeline, dali_dataset = get_pipe_dataset(
        batch_size,
        num_threads,
        device,
        device_id,
        get_pipeline_desc=get_pipeline_desc,
        to_dataset=to_dataset,
    )

    dataset_results = run_dataset_in_graph(dali_dataset, iterations, to_stop_iter=to_stop_iter)
    standalone_results = run_pipeline(
        standalone_pipeline, iterations, device, to_stop_iter=to_stop_iter
    )

    compare(dataset_results, standalone_results)


def run_tf_dataset_eager_mode(
    device,
    device_id=0,
    get_pipeline_desc=get_image_pipeline,
    to_dataset=to_image_dataset,
    to_stop_iter=False,
):
    batch_size = 12
    num_threads = 4
    iterations = 10

    standalone_pipeline, dali_dataset = get_pipe_dataset(
        batch_size,
        num_threads,
        device,
        device_id,
        get_pipeline_desc=get_pipeline_desc,
        to_dataset=to_dataset,
    )

    dataset_results = run_dataset_eager_mode(dali_dataset, iterations, to_stop_iter=to_stop_iter)
    standalone_results = run_pipeline(
        standalone_pipeline, iterations, device, to_stop_iter=to_stop_iter
    )

    compare(dataset_results, standalone_results)


def run_tf_dataset_multigpu_graph_manual_placement(
    get_pipeline_desc=get_image_pipeline, to_dataset=to_image_dataset
):
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    dali_datasets = []
    standalone_pipelines = []
    for device_id in range(num_devices):
        standalone_pipeline, dali_dataset = get_pipe_dataset(
            batch_size,
            num_threads,
            "gpu",
            device_id,
            num_devices,
            get_pipeline_desc=get_pipeline_desc,
            to_dataset=to_dataset,
            dali_on_dev_0=False,
        )
        dali_datasets.append(dali_dataset)
        standalone_pipelines.append(standalone_pipeline)

    dataset_results = run_dataset_in_graph(dali_datasets, iterations)
    standalone_results = run_pipeline(standalone_pipelines, iterations, "gpu")

    compare(dataset_results, standalone_results, iterations, num_devices)


def run_tf_dataset_multigpu_eager_manual_placement(
    get_pipeline_desc=get_image_pipeline, to_dataset=to_image_dataset
):
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    dali_datasets = []
    standalone_pipelines = []
    for device_id in range(num_devices):
        standalone_pipeline, dali_dataset = get_pipe_dataset(
            batch_size,
            num_threads,
            "gpu",
            device_id,
            num_devices,
            get_pipeline_desc=get_pipeline_desc,
            to_dataset=to_dataset,
            dali_on_dev_0=False,
        )
        dali_datasets.append(dali_dataset)
        standalone_pipelines.append(standalone_pipeline)

    dataset_results = run_dataset_eager_mode(dali_datasets, iterations)
    standalone_results = run_pipeline(standalone_pipelines, iterations, "gpu")

    compare(dataset_results, standalone_results, iterations, num_devices)


def per_replica_to_numpy(dataset_results, num_devices):
    results = []
    for sample in dataset_results:
        new_sample = []
        for device_id in range(num_devices):
            new_batch = []
            for output in range(len(sample[0])):
                new_batch.append(sample[0][output].values[device_id].numpy())
            new_sample.append(new_batch)
        results.append(new_sample)
    return results


def run_tf_dataset_multigpu_eager_mirrored_strategy(
    get_pipeline_desc=get_image_pipeline, to_dataset=to_image_dataset
):
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    strategy = tf.distribute.MirroredStrategy(devices=available_gpus())
    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_fetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
    )

    def dataset_fn(input_context):
        return get_dali_dataset(
            batch_size,
            num_threads,
            "gpu",
            input_context.input_pipeline_id,
            num_devices,
            get_pipeline_desc=get_pipeline_desc,
            to_dataset=to_dataset,
        )

    dali_datasets = [strategy.distribute_datasets_from_function(dataset_fn, input_options)]
    dataset_results = run_dataset_eager_mode(dali_datasets, iterations)

    standalone_pipelines = []
    for device_id in range(num_devices):
        pipeline, _, _ = get_pipeline_desc(
            batch_size,
            num_threads,
            device="gpu",
            device_id=device_id,
            shard_id=device_id,
            num_shards=num_devices,
        )
        standalone_pipelines.append(pipeline)

    standalone_results = run_pipeline(standalone_pipelines, iterations, "gpu")

    dataset_results = per_replica_to_numpy(dataset_results, num_devices)
    compare(dataset_results, standalone_results, iterations, num_devices)
