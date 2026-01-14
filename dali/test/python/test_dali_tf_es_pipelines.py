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

import copy
import numpy as np
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from nvidia.dali import Pipeline, fn

from test_utils import RandomlyShapedDataIterator


def get_sample_one_arg_callback(dtype, iter_limit=1000, batch_size=None, dense=True):
    def callback(x):
        if x.iteration > iter_limit:
            raise StopIteration()
        size = x.idx_in_batch % 16 + 1, x.iteration % 16 + 4
        result = np.full(size, x.idx_in_epoch, dtype=dtype)
        result[0][0] = x.idx_in_epoch
        result[0][1] = x.idx_in_batch
        result[0][2] = x.iteration
        result[0][3] = x.epoch_idx
        return result

    return callback


def get_batch_one_arg_callback(dtype, iter_limit=1000, batch_size=None, dense=True):
    def callback(x):
        if x > iter_limit:
            raise StopIteration()
        size = (x % 16 + 3,)
        result = [np.full(size, x, dtype=dtype)] * batch_size
        for i, elem in enumerate(result):
            elem[0] = i
            elem[1] = x
        return np.stack(result) if dense else result

    return callback


def get_batch_one_arg_callback_with_batch_info(dtype, iter_limit=1000, batch_size=None, dense=True):
    def callback(x):
        if x.iteration > iter_limit:
            raise StopIteration()
        size = (x.iteration % 16 + 4,)
        result = [np.full(size, x.iteration, dtype=dtype)] * batch_size
        for i, elem in enumerate(result):
            elem[0] = i
            elem[1] = x.iteration
            elem[2] = x.epoch_idx
        return np.stack(result) if dense else result

    return callback


def get_no_arg_callback(dtype, iter_limit=1000, batch_size=None, dense=True):
    class Callable:
        def __init__(self):
            self.counter = 0

        def __call__(self):
            size = (self.counter % 16 + 3,)
            bs = 1 if batch_size is None else batch_size
            if self.counter // bs > iter_limit:
                self.counter = 0
                raise StopIteration()
            curr_counter = self.counter
            self.counter += 1
            if batch_size is None:
                result = np.full(size, curr_counter, dtype=dtype)
                return result
            else:
                result = [np.full(size, curr_counter, dtype=dtype)] * batch_size
                for i, elem in enumerate(result):
                    elem[0] = i
                return np.stack(result)

    return Callable()


class UnwrapIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)[0]


class DenseIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return np.stack(next(self.iterator))


class FiniteIterator:
    """Used to wrap RandomlyShapedDataIterator to add iteration counts and finite data size"""

    def __init__(self, iterator, iter_limit):
        self.iterator = iterator
        self.iter_limit = iter_limit

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.iter_limit:
            raise StopIteration()
        result = next(self.iterator)
        for i, elem in enumerate(result):
            assert len(elem.shape) == (2), f"Got unexpected shape {elem.shape}"
            assert elem.shape[1] >= 2, f"Got unexpected shape {elem.shape}"
            elem[0][0] = i
            elem[0][1] = self.i
        self.i += 1
        return result


def get_iterable(dtype, iter_limit=1000, batch_size=None, dense=True):
    bs = 1 if batch_size is None else batch_size
    max_shape = (20, 20)
    min_shape = max_shape  # if dense else None
    result = FiniteIterator(
        iter(RandomlyShapedDataIterator(bs, min_shape, max_shape, 42, dtype)), iter_limit
    )
    if batch_size is None:
        return UnwrapIterator(iter(result))
    else:
        return DenseIterator(iter(result)) if dense else result


def get_iterable_generator(dtype, iter_limit=1000, batch_size=None, dense=True):
    def generator():
        iterator = iter(get_iterable(dtype, iter_limit, batch_size, dense))
        for example in iterator:
            yield example

    return generator


# generator, is_batched, cycle, batch_info
# TODO(klecki): cycle='raise' is currently not supported, and probably never will be
es_configurations = [
    (get_sample_one_arg_callback, False, None, False),
    (get_batch_one_arg_callback, True, None, False),
    (get_batch_one_arg_callback_with_batch_info, True, None, True),
    (get_no_arg_callback, False, None, False),
    (get_no_arg_callback, True, None, False),
    (get_iterable, False, False, False),
    (get_iterable, False, True, False),
    # (get_iterable, False, "raise", False),
    (get_iterable, True, False, False),
    (get_iterable, True, True, False),
    # (get_iterable, True, "raise", False),
    (get_iterable_generator, False, False, False),
    (get_iterable_generator, False, True, False),
    # (get_iterable_generator, False, "raise", False),
    (get_iterable_generator, True, False, False),
    (get_iterable_generator, True, True, False),
    # (get_iterable_generator, True, "raise", False),
]


def get_external_source_pipe(es_args, dtype, es_device):
    def get_pipeline_desc(
        batch_size, num_threads, device, device_id, shard_id, num_shards, def_for_dataset
    ):
        pipe = Pipeline(batch_size, num_threads, device_id)
        with pipe:
            # Our callbacks may have state, to be able to run it twice, once in Dataset and once
            # with baseline test, we need to make a copy to preserve that state.
            es = fn.external_source(device=es_device, **copy.deepcopy(es_args))
            if device == "gpu" and es_device == "cpu":
                es = es.gpu()
            pad = fn.pad(es, device=device)
            pipe.set_outputs(pad)
        return pipe, None, dtype

    return get_pipeline_desc


def external_source_to_tf_dataset(pipe_desc, device_str):  # -> tf.data.Dataset
    pipe, _, dtypes = pipe_desc
    with tf.device(device_str):
        dali_dataset = dali_tf.experimental.DALIDatasetWithInputs(
            input_datasets=None,
            pipeline=pipe,
            batch_size=pipe.max_batch_size,
            output_shapes=None,
            output_dtypes=dtypes,
            num_threads=pipe.num_threads,
            device_id=pipe.device_id,
        )
    return dali_dataset


def get_dense_options(is_batched):
    if is_batched:
        return [True, False]
    else:
        return [True]


def gen_tf_with_dali_external_source(test_run):
    for dtype in [np.uint8, np.int32, np.float32]:
        for get_callback, is_batched, cycle, batch_info in es_configurations:
            for dense in get_dense_options(is_batched):
                for dev, es_dev in [("cpu", "cpu"), ("gpu", "cpu"), ("gpu", "gpu")]:
                    for iter_limit in [3, 9, 10, 11, 100]:
                        bs = 12 if is_batched else None
                        es_args = {
                            "source": get_callback(dtype, iter_limit, bs, dense),
                            "batch": is_batched,
                            "cycle": cycle,
                            "batch_info": batch_info,
                        }
                        yield test_run, dev, es_args, es_dev, tf.dtypes.as_dtype(
                            dtype
                        ), iter_limit, dense
