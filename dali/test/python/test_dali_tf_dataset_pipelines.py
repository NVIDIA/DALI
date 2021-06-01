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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali import pipeline_def


import tensorflow as tf
from test_utils import  RandomlyShapedDataIterator

import numpy as np


from nose.tools import nottest

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


@pipeline_def
def one_input_pipeline(def_for_dataset, device, source):
    if def_for_dataset:
        # TODO(klecki): We need a way to control definition of DALI Pipeline, to set
        # the appropriate no_copy (and other options), maybe we can do it in Python by inspecting
        # the defined graph?
        input = fn.external_source(name="input_placeholder", no_copy=device == "cpu")
    else:
        input = fn.external_source(name="actual_input", source=source, batch=False)
    input = input if device == 'cpu' else input.gpu()
    processed = fn.cast(input + 10, dtype=dali.types.INT32)
    input_padded, processed_padded = fn.pad([input, processed])
    return input_padded, processed_padded


def external_source_converter_with_fixed_value(shape, dtype, tensor):
    def to_dataset(pipeline_desc, device_str):
        with tf.device('/cpu:0'):
            input_dataset = tf.data.Dataset.from_tensors(tensor).repeat()
            # If we place DALIDataset on GPU we need the remote call + manual data transfer
            if "gpu" in device_str:
                input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))


        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device(device_str):
            dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                    input_datasets=input_dataset,
                    input_names="input_placeholder",
                    pipeline=dataset_pipeline,
                    batch_size=dataset_pipeline.batch_size,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    num_threads=dataset_pipeline.num_threads,
                    device_id=dataset_pipeline.device_id)
        return dali_dataset
    return to_dataset

def external_source_converter_with_callback(shape, dtype, input_iterator):
    def to_dataset(pipeline_desc, device_str):
        with tf.device('/cpu:0'):
            out_shape = tuple(None for _ in shape)
            tf_type = tf.dtypes.as_dtype(dtype)
            input_dataset = tf.data.Dataset.from_generator(
                input_iterator, output_types=tf_type, output_shapes=out_shape, args=(shape, dtype(0)))
            # If we place DALIDataset on GPU we need the remote call + manual data transfer
            if "gpu" in device_str:
                input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))

        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device(device_str):
            dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                    input_datasets=input_dataset,
                    input_names="input_placeholder",
                    pipeline=dataset_pipeline,
                    batch_size=dataset_pipeline.batch_size,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    num_threads=dataset_pipeline.num_threads,
                    device_id=dataset_pipeline.device_id)
        return dali_dataset
    return to_dataset

@nottest
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
