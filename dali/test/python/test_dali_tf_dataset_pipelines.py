# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from posix import SEEK_DATA
import nvidia.dali as dali

import nvidia.dali.plugin.tf as dali_tf
# import os
# import numpy as np
from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
# import nvidia.dali.types as types

import tensorflow as tf
from test_utils import RandomDataIterator, RandomlyShapedDataIterator

import numpy as np

# from tensorflow.python.client import device_lib
# from nose import SkipTest
# from distutils.version import LooseVersion
# import math

def get_dali_dataset_from_pipeline_in(dataset_pipeline, batch_size, num_threads, device, device_id, num_devices=1):
    shapes = ((batch_size, None), (batch_size, None))
    dtypes = (tf.int32, tf.int32,)

    # dataset = tf.data.Dataset.from_tensors([1, 2, 3])

    # class generator:

    #     def __init__(self, max=0):
    #         self.max = max

    #     def __iter__(self):
    #         self.n = 0
    #         return self

    #     def __next__(self):
    #         if self.n <= self.max:
    #             result = self.n
    #             self.n += 1
    #             return np.full([12, 2], self.n, dtype=np.int32), np.full([12, 3, 4], self.n, dtype=np.int32)
    #         else:
    #             raise StopIteration

    # dataset = tf.data.Dataset.from_generator(
    #     generator, output_types=(tf.int32, tf.int32), output_shapes=((12, 2), (12, 3, 4)), args=(100,))



    # TF is very vocal of difference between `tf.int32` and `(tf.int32,)`
    dataset = tf.data.Dataset.from_generator(
        simple_generator, output_types=tf.int32, output_shapes=(None,), args=(8,))


    with tf.device('/{0}:{1}'.format(device, device_id)):
        dali_dataset = dali_tf.DALIDataset(
            input_datasets=dataset,
            input_names="ex_in_0",
            pipeline=dataset_pipeline,
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            num_threads=num_threads,
            device_id=device_id)
    return dali_dataset.repeat()

# def get_external_source_pipeline_getter(batch_size, num_threads, device, device_id=0, shard_id=0, num_shards=1):


class RandomSampleIterator:
    def __init__(self, max_shape=(10, 600, 800, 3), dtype_sample=np.uint8(0), min_shape=None, seed=42, start=0, stop=10e100):
        self.start = start
        self.stop = stop
        self.min_shape = min_shape
        self.max_shape = max_shape
        # As tf passes only tensors to the iterator, we pass a dummy value of which we take the type
        self.dtype = dtype_sample.dtype
        self.seed = seed
        print("RSI init")

    def __iter__(self):
        print("RSI iter")
        self.n = self.start
        self.random_iter = iter(RandomlyShapedDataIterator(batch_size=1, min_shape=self.min_shape,
                max_shape=self.max_shape, seed=self.seed, dtype=self.dtype))
        return self

    def __next__(self):
        print("RSI next")
        if self.n <= self.stop:
            result = self.n
            self.n += 1
            ret = self.random_iter.next()[0]
            return ret
        else:
            raise StopIteration


class FixedSampleIterator:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return self

    def __next__(self):
        return self.value




# def external_source_cpu_to_gpu(shape, dtype):
#     # create it out of the DALI Dataset placement scope

#     # RandomSampleIterator(max_shape=shape, dtype=dtype)

#     def to_dataset(pipeline_desc, device_str):
#         with tf.device('/cpu:0'):
#             out_shape = tuple(None for _ in shape)
#             # input_dataset = tf.data.Dataset.from_generator(
#             #     RandomSampleIterator, output_types=tf.dtypes.as_dtype(dtype), output_shapes=out_shape, args=(shape,))

#             # input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/cpu:0'))
#             input_dataset = tf.data.Dataset.from_tensors(np.full((10, 20), 42, dtype=np.uint8)).repeat()

#         print(input_dataset)

#         dataset_pipeline, shapes, dtypes = pipeline_desc

#         with tf.device(device_str):
#             print("creating daliDataset")
#             dali_dataset = dali_tf.DALIDataset(
#                     input_datasets=input_dataset,
#                     input_names="input_placeholder",
#                     pipeline=dataset_pipeline,
#                     batch_size=dataset_pipeline.batch_size,
#                     output_shapes=shapes,
#                     output_dtypes=dtypes,
#                     num_threads=dataset_pipeline.num_threads,
#                     device_id=dataset_pipeline.device_id)
#         return dali_dataset
#     return to_dataset


# def external_source_converter_gpu(shape, dtype):
#     # create it out of the DALI Dataset placement scope

#     # RandomSampleIterator(max_shape=shape, dtype=dtype)

#     def to_dataset(pipeline_desc, device_str):
#         with tf.device('/cpu:0'):
#             out_shape = tuple(None for _ in shape)
#             # input_dataset = tf.data.Dataset.from_generator(
#             #     RandomSampleIterator, output_types=tf.dtypes.as_dtype(dtype), output_shapes=out_shape, args=(shape,))
#             input_dataset = tf.data.Dataset.from_tensors(np.full((10, 20), 42, dtype=np.uint8)).repeat()
#             input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))

#         print(input_dataset)

#         dataset_pipeline, shapes, dtypes = pipeline_desc
#         with tf.device(device_str):
#             print("creating daliDataset")
#             dali_dataset = dali_tf.DALIDataset(
#                     input_datasets=input_dataset,
#                     input_names="input_placeholder",
#                     input_devices="gpu",
#                     pipeline=dataset_pipeline,
#                     batch_size=dataset_pipeline.batch_size,
#                     output_shapes=shapes,
#                     output_dtypes=dtypes,
#                     num_threads=dataset_pipeline.num_threads,
#                     device_id=dataset_pipeline.device_id)
#         return dali_dataset
#     return to_dataset

@pipeline_def
def one_input_pipeline(def_for_dataset, device, source):
    if def_for_dataset:
        input = fn.external_source(name="input_placeholder")
    else:
        input = fn.external_source(name="actual_input", source=source, batch=False)
    input = input if device == 'cpu' else input.gpu()
    processed = fn.cast(input + 10, dtype=dali.types.INT32)
    input_padded, processed_padded = fn.pad([input, processed])
    return input_padded, processed_padded


def external_source_converter_with_fixed_value(shape, dtype, tensor, input_device='/cpu:0'):
    def to_dataset(pipeline_desc, device_str):
        with tf.device(input_device):
            input_dataset = tf.data.Dataset.from_tensors(tensor).repeat().map(lambda x: x+1)
            # input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))


        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device(device_str):
            dali_dataset = dali_tf.DALIDataset(
                    input_datasets=input_dataset,
                    input_names="input_placeholder",
                    input_devices="cpu",
                    pipeline=dataset_pipeline,
                    batch_size=dataset_pipeline.batch_size,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    num_threads=dataset_pipeline.num_threads,
                    device_id=dataset_pipeline.device_id)
        return dali_dataset
    return to_dataset

def external_source_converter_with_callback(shape, dtype, input_iterator, input_device='/cpu:0'):
    def to_dataset(pipeline_desc, device_str):
        with tf.device(input_device):
            out_shape = tuple(None for _ in shape)
            tf_type = tf.dtypes.as_dtype(dtype)
            input_dataset = tf.data.Dataset.from_generator(
                input_iterator, output_types=tf_type, output_shapes=out_shape, args=(shape, dtype(0)))
            # we need the remote call for GPU dataset in this particular case
            input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/cpu:0'))

        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device(device_str):
            dali_dataset = dali_tf.DALIDataset(
                    input_datasets=input_dataset,
                    input_names="input_placeholder",
                    input_devices="cpu" if device_str == "/cpu:0" else "gpu",
                    pipeline=dataset_pipeline,
                    batch_size=dataset_pipeline.batch_size,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    num_threads=dataset_pipeline.num_threads,
                    device_id=dataset_pipeline.device_id)
        return dali_dataset
    return to_dataset


def external_source_tester(shape, dtype, source=None):
    def get_external_source_pipeline_getter(batch_size, num_threads, device, device_id=0,
        shard_id=0, num_shards=1, def_for_dataset=False):

        pipe = one_input_pipeline(def_for_dataset,
                                  device,
                                  source,
                                  batch_size=batch_size,
                                  num_threads=num_threads,
                                  device_id=device_id)

        batch_shape = (batch_size,) + tuple(None for _ in shape)

        return pipe, (batch_shape, batch_shape), (tf.dtypes.as_dtype(dtype), tf.int32)
    return get_external_source_pipeline_getter


# def get_dali_dataset_in(batch_size, num_threads, device, device_id, num_devices=1):
#     shard_id = 0 if num_devices == 1 else device_id
#     dataset_pipeline = get_pipeline_in(
#         batch_size, num_threads, device, device_id, shard_id, num_devices)

#     return get_dali_dataset_from_pipeline_in(
#         dataset_pipeline, batch_size, num_threads, device, device_id, num_devices)

# def run_dataset_eager_mode_in(dali_datasets, iterations):
#     if type(dali_datasets) is not list:
#         dali_datasets = [dali_datasets]

#     results = []
#     for i, batch in zip(range(iterations), zip(*dali_datasets)):
#         results.append(batch)
#     return results



# def run_tf_dataset_eager_mode_in(device, device_id=0):
#     batch_size = 2
#     num_threads = 4
#     iterations = 10

#     dali_dataset = get_dali_dataset_in(batch_size, num_threads, device, device_id)
#     dataset_results = run_dataset_eager_mode_in(dali_dataset, iterations)


#     # standalone_pipeline = get_pipeline(
#     #     batch_size, num_threads, device, device_id=0)
#     # standalone_results = run_pipeline(standalone_pipeline, iterations, device)
#     for it in range(len(dataset_results)):
#         print (" >> DATASET OUTPUT {} <<".format(it))
#         sample = dataset_results[it]
#         print(sample)
#     # compare(dataset_results, standalone_results)