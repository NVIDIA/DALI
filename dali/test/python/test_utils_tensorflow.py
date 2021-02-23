# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import nvidia.dali as dali

import nvidia.dali.plugin.tf as dali_tf
import os
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import tensorflow as tf
from tensorflow.python.client import device_lib
from nose import SkipTest
from distutils.version import LooseVersion
import math


def skip_for_incompatible_tf():
    if not dali_tf.dataset_distributed_compatible_tensorflow():
        raise SkipTest('This feature is enabled for TF 2.5.0 and higher')


def num_available_gpus():
    local_devices = device_lib.list_local_devices()
    num_gpus = sum(1 for device in local_devices if device.device_type == 'GPU')
    if not math.log2(num_gpus).is_integer():
        raise RuntimeError('Unsupported number of GPUs. This test can run on: 2^n GPUs.')
    return num_gpus


def available_gpus():
    devices = []
    for device_id in range(num_available_gpus()):
        devices.append('/gpu:{0}'.format(device_id))
    return devices


def get_pipeline(batch_size, num_threads, device, device_id=0, shard_id=0, num_shards=1):
    test_data_root = os.environ['DALI_EXTRA_PATH']
    file_root = os.path.join(test_data_root, 'db', 'coco_dummy', 'images')
    annotations_file = os.path.join(
        test_data_root, 'db', 'coco_dummy', 'instances.json')

    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, _, _, image_ids = fn.readers.coco(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=False,
            image_ids=True)
        images = fn.image_decoder(
            jpegs,
            device=('mixed' if device == 'gpu' else 'cpu'),
            output_type=types.RGB)
        images = fn.resize(
            images,
            resize_x=224,
            resize_y=224,
            interp_type=types.INTERP_LINEAR)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            mean=[128., 128., 128.],
            std=[1., 1., 1.])
        if device == 'gpu':
            image_ids = image_ids.gpu()
        ids_reshaped = fn.reshape(image_ids, shape=[1, 1])
        ids_int16 = fn.cast(image_ids, dtype=types.INT16)

        pipe.set_outputs(images, ids_reshaped, ids_int16)

    return pipe


def get_dali_dataset_from_pipeline(dataset_pipeline, batch_size, num_threads, device, device_id, num_devices=1):
    shapes = (
        (batch_size, 3, 224, 224),
        (batch_size, 1, 1),
        (batch_size, 1))
    dtypes = (
        tf.float32,
        tf.int32,
        tf.int16)

    with tf.device('/{0}:{1}'.format(device, device_id)):
        dali_dataset = dali_tf.DALIDataset(
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            num_threads=num_threads,
            device_id=device_id)
    return dali_dataset


def get_dali_dataset(batch_size, num_threads, device, device_id, num_devices=1):
    shard_id = 0 if num_devices == 1 else device_id
    dataset_pipeline = get_pipeline(
        batch_size, num_threads, device, device_id, shard_id, num_devices)

    return get_dali_dataset_from_pipeline(
        dataset_pipeline, batch_size, num_threads, device, device_id, num_devices)


def run_dataset_in_graph(dali_datasets, iterations):
    if type(dali_datasets) is not list:
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
        for _ in range(iterations):
            dataset_results.append(sess.run(ops_to_run))
    return dataset_results


def run_dataset_eager_mode(dali_datasets, iterations):
    if type(dali_datasets) is not list:
        dali_datasets = [dali_datasets]

    results = []
    for i, batch in zip(range(iterations), zip(*dali_datasets)):
        results.append(batch)
    return results


def run_pipeline(pipeline, iterations, device):
    pipeline.build()
    results = []
    for _ in range(iterations):
        if device == 'gpu':
            results.append(
                tuple(result.as_cpu().as_array() for result in pipeline.run()))
        else:
            results.append(
                tuple(result.as_array() for result in pipeline.run()))
    return results


def compare(dataset_results, standalone_results, iterations=-1,  num_devices=1):
    if iterations == -1:
        iterations = len(standalone_results)

    assert len(dataset_results) == iterations
    for it in range(iterations):
        sample = dataset_results[it]
        standalone_sample = standalone_results[it]
        assert len(dataset_results[it]) == num_devices
        for device_id in range(num_devices):
            batch_id = iterations - \
                ((it + device_id * (iterations // num_devices)) % iterations) - 1
            it_id = iterations - it - 1
            assert np.array_equal(
                standalone_results[it_id][0],
                dataset_results[batch_id][device_id][0])
            assert np.array_equal(
                standalone_results[it_id][1],
                dataset_results[batch_id][device_id][1])
            assert np.array_equal(
                standalone_results[it_id][2],
                dataset_results[batch_id][device_id][2])


def run_tf_dataset_graph(device, device_id=0):
    batch_size = 12
    num_threads = 4
    iterations = 10

    dali_dataset = get_dali_dataset(batch_size, num_threads, device, device_id)
    dataset_results = run_dataset_in_graph(dali_dataset, iterations)

    standalone_pipeline = get_pipeline(
        batch_size, num_threads, device, device_id=0)
    standalone_results = run_pipeline(standalone_pipeline, iterations, device)

    compare(dataset_results, standalone_results)


def run_tf_dataset_eager_mode(device, device_id=0):
    batch_size = 12
    num_threads = 4
    iterations = 10

    dali_dataset = get_dali_dataset(batch_size, num_threads, device, device_id)
    dataset_results = run_dataset_eager_mode(dali_dataset, iterations)

    standalone_pipeline = get_pipeline(
        batch_size, num_threads, device, device_id=0)
    standalone_results = run_pipeline(standalone_pipeline, iterations, device)

    compare(dataset_results, standalone_results)

def run_tf_dataset_multigpu_graph_manual_placement():
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    dali_datasets = []
    for device_id in range(num_devices):
        dali_datasets.append(get_dali_dataset(
            batch_size, num_threads, 'gpu', device_id, num_devices))

    dataset_results = run_dataset_in_graph(dali_datasets, iterations)

    standalone_pipeline = get_pipeline(
        batch_size, num_threads, device='gpu', device_id=0)
    standalone_results = run_pipeline(standalone_pipeline, iterations, 'gpu')

    compare(dataset_results, standalone_results, iterations, num_devices)


def run_tf_dataset_multigpu_eager_manual_placement():
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    dali_datasets = []
    for device_id in range(num_devices):
        dali_datasets.append(get_dali_dataset(
            batch_size, num_threads, 'gpu', device_id, num_devices))

    dataset_results = run_dataset_eager_mode(dali_datasets, iterations)

    standalone_pipeline = get_pipeline(
        batch_size, num_threads, device='gpu', device_id=0)
    standalone_results = run_pipeline(standalone_pipeline, iterations, 'gpu')

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


def run_tf_dataset_multigpu_eager_mirrored_strategy():
    num_devices = num_available_gpus()
    batch_size = 8
    num_threads = 4
    iterations = 8

    strategy = tf.distribute.MirroredStrategy(devices=available_gpus())
    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device = True,
        experimental_prefetch_to_device = False,
        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)

    def dataset_fn(input_context):
        return get_dali_dataset(
            batch_size, num_threads, 'gpu', input_context.input_pipeline_id, num_devices)

    dali_datasets = [
        strategy.distribute_datasets_from_function(dataset_fn, input_options)]
    dataset_results = run_dataset_eager_mode(dali_datasets, iterations)

    standalone_pipeline = get_pipeline(
        batch_size, num_threads, device='gpu', device_id=0)
    standalone_results = run_pipeline(standalone_pipeline, iterations, 'gpu')

    dataset_results = per_replica_to_numpy(dataset_results, num_devices)
    compare(dataset_results, standalone_results, iterations, num_devices)
