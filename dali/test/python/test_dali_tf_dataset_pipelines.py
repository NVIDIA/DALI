# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from nose_utils import nottest
from nvidia.dali import pipeline_def

from test_utils import RandomlyShapedDataIterator


def get_min_shape_helper(batch, max_shape):
    """For batch=None or batch=True, we use batch mode which requires fixed shape.
    In that case min and max shape for RandomSampleIterator need to be equal.
    `batch` can also be a string "dataset" that indicates we passed a Dataset object as input
    without specifying the batch mode through: Input(dataset, batch=...)
    """
    if batch is None or batch is True:
        return max_shape
    else:
        return None


class RandomSampleIterator:
    def __init__(
        self,
        max_shape=(10, 600, 800, 3),
        dtype_sample=np.uint8(0),
        start=0,
        stop=1e100,
        min_shape=None,
        seed=42,
    ):
        self.start = start
        self.stop = stop
        self.min_shape = min_shape
        self.max_shape = max_shape
        # As tf passes only tensors to the iterator, we pass a dummy value of which we take the type
        self.dtype = dtype_sample.dtype
        self.seed = seed

    def __iter__(self):
        self.n = self.start
        self.random_iter = iter(
            RandomlyShapedDataIterator(
                batch_size=1,
                min_shape=self.min_shape,
                max_shape=self.max_shape,
                seed=self.seed,
                dtype=self.dtype,
            )
        )
        return self

    def __next__(self):
        if self.n <= self.stop:
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


class InfiniteSampleIterator:
    def __init__(self, start_value):
        self.value = start_value

    def __iter__(self):
        return self

    def __next__(self):
        result = self.value
        self.value = self.value + np.array(1, dtype=self.value.dtype)
        return result


@pipeline_def
def one_input_pipeline(def_for_dataset, device, source, external_source_device, no_copy, batch):
    """Pipeline accepting single input via external source

    Parameters
    ----------
    def_for_dataset : bool
         True if this pipeline will be converted to TF Dataset
    device : str
        device that the Dataset will be placed ("cpu" or "gpu")
    source : callable
        callback for the external source in baseline pipeline otherwise None
    external_source_device : str
        Device that we want the external source in TF dataset to be placed
    """

    if def_for_dataset:
        if no_copy is None:
            # If no_copy is None, we infer it automatically and we use no_copy=True when
            # the input memory is matching the external source placement,
            # so the Dataset's placement is the same as external source's device,
            # otherwise for cross-backend we use False.
            no_copy = device == external_source_device
        if batch == "dataset":
            # Special value used in tests, reroute it to the default
            batch = None
        input = fn.external_source(
            name="input_placeholder", no_copy=no_copy, device=external_source_device, batch=batch
        )
    else:
        input = fn.external_source(
            name="actual_input", source=source, batch=False, device=external_source_device
        )
    input = input if device == "cpu" else input.gpu()
    processed = fn.cast(input + 10, dtype=dali.types.INT32)
    input_padded, processed_padded = fn.pad([input, processed])
    return input_padded, processed_padded


# Test that uses Tensor and Repeat (infinite) datasets as inputs to DALI pipeline
def external_source_converter_with_fixed_value(shape, dtype, tensor, batch="dataset"):
    def to_dataset(pipeline_desc, device_str):
        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device("/cpu:0"):
            input_dataset = tf.data.Dataset.from_tensors(tensor).repeat()
            if batch is None or batch is True:
                input_dataset = input_dataset.batch(dataset_pipeline.max_batch_size)
            # If we place DALIDataset on GPU we need the remote call + manual data transfer
            if "gpu" in device_str:
                input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))

        if batch == "dataset":
            input_datasets = {"input_placeholder": input_dataset}
        else:
            input_datasets = {
                "input_placeholder": dali_tf.experimental.Input(input_dataset, batch=batch)
            }

        with tf.device(device_str):
            dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                input_datasets=input_datasets,
                pipeline=dataset_pipeline,
                batch_size=dataset_pipeline.max_batch_size,
                output_shapes=shapes,
                output_dtypes=dtypes,
                num_threads=dataset_pipeline.num_threads,
                device_id=dataset_pipeline.device_id,
            )
        return dali_dataset

    return to_dataset


def external_source_converter_with_callback(
    input_iterator,
    shape,
    dtype,
    start_samples=0,
    stop_samples=1e10,
    min_shape=None,
    batch="dataset",
):
    """Test that uses Generator dataset as inputs to DALI pipeline"""

    def to_dataset(pipeline_desc, device_str):
        dataset_pipeline, shapes, dtypes = pipeline_desc

        with tf.device("/cpu:0"):
            _args = (shape, dtype(0), start_samples, stop_samples)
            # Add min_shape if it's not None
            _args = _args + ((min_shape,) if min_shape is not None else ())
            out_shape = tuple(None for _ in shape)
            tf_type = tf.dtypes.as_dtype(dtype)
            input_dataset = tf.data.Dataset.from_generator(
                input_iterator, output_types=tf_type, output_shapes=out_shape, args=_args
            )
            if batch is None or batch is True:
                input_dataset = input_dataset.batch(dataset_pipeline.max_batch_size)
            # If we place DALIDataset on GPU we need the remote call + manual data transfer
            if "gpu" in device_str:
                input_dataset = input_dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))

        if batch == "dataset":
            input_datasets = {"input_placeholder": input_dataset}
        else:
            input_datasets = {
                "input_placeholder": dali_tf.experimental.Input(input_dataset, batch=batch)
            }

        with tf.device(device_str):
            dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                input_datasets=input_datasets,
                pipeline=dataset_pipeline,
                batch_size=dataset_pipeline.max_batch_size,
                output_shapes=shapes,
                output_dtypes=dtypes,
                num_threads=dataset_pipeline.num_threads,
                device_id=dataset_pipeline.device_id,
            )
        return dali_dataset

    return to_dataset


@nottest
def external_source_tester(
    shape, dtype, source=None, external_source_device="cpu", no_copy=None, batch=False
):
    def get_external_source_pipeline_getter(
        batch_size,
        num_threads,
        device,
        device_id=0,
        shard_id=0,
        num_shards=1,
        def_for_dataset=False,
    ):
        pipe = one_input_pipeline(
            def_for_dataset,
            device,
            source,
            external_source_device,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            no_copy=no_copy,
            batch=batch,
        )

        batch_shape = (batch_size,) + tuple(None for _ in shape)

        return pipe, (batch_shape, batch_shape), (tf.dtypes.as_dtype(dtype), tf.int32)

    return get_external_source_pipeline_getter


@pipeline_def
def many_input_pipeline(def_for_dataset, device, sources, input_names, batches):
    """Pipeline accepting multiple inputs via external source

    Parameters
    ----------
    def_for_dataset : bool
         True if this pipeline will be converted to TF Dataset
    device : str
        device that the Dataset will be placed ("cpu" or "gpu")
    sources : list of callables
        callbacks for the external sources in baseline pipeline otherwise None
    input_names : list of str
        Names of inputs placeholder for TF
    """
    inputs = []
    if def_for_dataset:
        for input_name, batch in zip(input_names, batches):
            if batch == "dataset":
                # Special value used in tests, reroute it to the default
                batch = None
            input = fn.external_source(name=input_name, batch=batch)
            input = input if device == "cpu" else input.gpu()
            inputs.append(input)
    else:
        for source in sources:
            input = fn.external_source(source=source, batch=False)
            input = input if device == "cpu" else input.gpu()
            inputs.append(input)
    processed = []
    for input in inputs:
        processed.append(fn.cast(input + 10, dtype=dali.types.INT32))
    results = fn.pad(inputs + processed)
    return tuple(results)


# Test that uses multiple Generator dataset as inputs to DALI pipeline
def external_source_converter_multiple(start_values, input_names, batches):
    def to_dataset(pipeline_desc, device_str):
        dataset_pipeline, shapes, dtypes = pipeline_desc

        input_datasets = {}
        with tf.device("/cpu:0"):
            for value, name, batch in zip(start_values, input_names, batches):
                tf_type = tf.dtypes.as_dtype(value.dtype)
                shape = value.shape
                input_dataset = tf.data.Dataset.from_generator(
                    InfiniteSampleIterator, output_types=tf_type, output_shapes=shape, args=(value,)
                )
                if batch is None or batch is True:
                    input_dataset = input_dataset.batch(dataset_pipeline.max_batch_size)
                # If we place DALIDataset on GPU we need the remote call + manual data transfer
                if "gpu" in device_str:
                    input_dataset = input_dataset.apply(
                        tf.data.experimental.copy_to_device("/gpu:0")
                    )
                if batch == "dataset":
                    input_datasets[name] = input_dataset
                else:
                    input_datasets[name] = dali_tf.experimental.Input(input_dataset, batch=batch)

        with tf.device(device_str):
            dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
                input_datasets=input_datasets,
                pipeline=dataset_pipeline,
                batch_size=dataset_pipeline.max_batch_size,
                output_shapes=shapes,
                output_dtypes=dtypes,
                num_threads=dataset_pipeline.num_threads,
                device_id=dataset_pipeline.device_id,
            )
        return dali_dataset

    return to_dataset


@nottest
def external_source_tester_multiple(start_values, input_names, batches):
    def get_external_source_pipeline_getter(
        batch_size,
        num_threads,
        device,
        device_id=0,
        shard_id=0,
        num_shards=1,
        def_for_dataset=False,
    ):
        sources = [InfiniteSampleIterator(start_value) for start_value in start_values]
        output_shapes = [
            ((batch_size,) + tuple(None for _ in start_value.shape)) for start_value in start_values
        ]
        output_shapes = tuple(output_shapes + output_shapes)
        output_dtypes = tuple(
            [tf.dtypes.as_dtype(start_value.dtype) for start_value in start_values]
            + [tf.int32] * len(start_values)
        )

        pipe = many_input_pipeline(
            def_for_dataset,
            device,
            sources,
            input_names,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            batches=batches,
        )

        return pipe, output_shapes, output_dtypes

    return get_external_source_pipeline_getter
