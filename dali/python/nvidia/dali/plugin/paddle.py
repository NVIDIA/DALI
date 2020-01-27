# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import logging
import math

import numpy as np

from nvidia.dali import types
from nvidia.dali.backend import TensorListCPU, TensorGPU, TensorListGPU
from paddle import fluid

dtype_map = {
    "=?": fluid.core.VarDesc.VarType.BOOL,
    "=e": fluid.core.VarDesc.VarType.FP16,
    "=f": fluid.core.VarDesc.VarType.FP32,
    "=d": fluid.core.VarDesc.VarType.FP64,
    "=B": fluid.core.VarDesc.VarType.UINT8,
    "=b": fluid.core.VarDesc.VarType.INT8,
    "=h": fluid.core.VarDesc.VarType.INT16,
    "=i": fluid.core.VarDesc.VarType.INT32,
    "=q": fluid.core.VarDesc.VarType.INT64,
    "=l": fluid.core.VarDesc.VarType.INT64
}


def to_paddle_type(tensor):
    r"""
    Get paddle dtype for given tensor or tensor list

    Args:
        tensor: tensor or tensor list

    Returns: fluid.core.VarDesc.VarType
    """
    if isinstance(tensor, (TensorListCPU, TensorListGPU)):
        tensor = tensor.at(0)
    dtype = tensor.dtype
    if callable(dtype):
        dtype = dtype()
    else:
        dtype = '=' + dtype.char
    return dtype_map[dtype]


def feed_ndarray(dali_tensor, ptr):
    """
    Copy contents of DALI tensor to Paddle's Tensor.

    Parameters
    ----------
    `dali_tensor` : dali.backend.TensorCPU or dali.backend.TensorGPU
                    Tensor from which to copy
    `ptr` : LoDTensor data pointer
            Destination of the copy
    """
    c_type_pointer = ctypes.c_void_p(ptr)
    dali_tensor.copy_to_external(c_type_pointer)
    return ptr


def recursive_length(tensor, lod_level):
    def _recurse(data, result, level):
        if level > 0:
            if isinstance(data, (TensorListCPU, TensorListGPU)):
                # handle tensor list
                length = len(data)
                result[0].append(length)
                for i in range(length):
                    _recurse(data.at(i), result[1:], level - 1)
            elif hasattr(data, 'shape'):
                # handle dense GPU tensors and numpy.ndarray
                shape = data.shape
                if callable(shape):
                    shape = shape()
                length = shape[0]
                result[0].append(length)
                for i in range(length):
                    _recurse(shape[1:], result[1:], level - 1)
            else:
                # handle shape
                length = data[0]
                result[0].append(length)
                for i in range(length):
                    _recurse(data[1:], result[1:], level - 1)

    seq_len = [[] for _ in range(lod_level)]
    _recurse(tensor, seq_len, lod_level)
    return seq_len


def lod_tensor_clip(lod_tensor, size):
    output = fluid.core.LoDTensor()
    ndarray = np.array(lod_tensor)
    seq_len = lod_tensor.recursive_sequence_lengths()
    if not seq_len:
        output.set(ndarray[0:size], fluid.CPUPlace())
    else:
        last_len = size
        out_seq_len = []
        for lengths in seq_len:
            lengths = lengths[0:last_len]
            out_seq_len.append(lengths)
            last_len = sum(lengths)
        output.set(ndarray[0:sum(out_seq_len[-1])], fluid.CPUPlace())
        output.set_recursive_sequence_lengths(out_seq_len)
    return output


class DALIGenericIterator(object):
    """
    General DALI iterator for Paddle. It can return any number of
    outputs from the DALI pipeline in the form of Paddle's Tensors.

    Please keep in mind that Tensors returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    output_map : list of str or pair of type (str, int)
                 The strings maps consecutive outputs of DALI pipelines to
                 user specified name. Outputs will be returned from iterator
                 as dictionary of those names. Each name should be distinct.
                 Item can also be a pair of (str, int), where the int value
                 specifies the LoD level of the resulting LoDTensor.
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to return a fraction of a full batch of data
                 such that the total entries returned by the
                 iterator == 'size'. Setting this flag to False will
                 cause the iterator to return the first integer multiple
                 of self._num_gpus * self.batch_size which exceeds 'size'.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the LoDTensor will be
                 resized accordingly if the shape of DALI returned tensors
                 changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one. For this happen, the option `pad_last_batch`
                 in the reader need to be set to `True` as well.

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    fill_last_batch = False, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    fill_last_batch = False, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    fill_last_batch = True, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    fill_last_batch = True, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 output_map,
                 size,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        assert pipelines is not None, \
            "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = int(size)
        self._auto_reset = auto_reset
        self._dynamic_shape = dynamic_shape
        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and len(pipelines) == 1), "Negative size is supported only for a single pipeline"
        if self._size < 0:
            self._auto_reset = False
            self._fill_last_batch = False
            self._last_batch_padded = False
        self._pipes = pipelines
        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()
        # Use double-buffering of data batches
        self._data_batches = [None for i in range(self._num_gpus)]
        self._counter = 0

        normalized_map = {}
        for v in output_map:
            if isinstance(v, str):
                normalized_map[v] = 0
            else:
                normalized_map[v[0]] = v[1]
        self.normalized_map = normalized_map

        output_map = [isinstance(v, str) and v or v[0] for v in output_map]
        assert len(set(output_map)) == len(output_map), \
            "output_map names should be distinct"
        self.output_map = output_map

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.schedule_run()
        self._first_batch = None
        self._first_batch = self.next()

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter >= self._size and self._size > 0:
            if self._auto_reset:
                self.reset()
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
               outputs.append(p.share_outputs())

        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            # Initialize dict for all output categories
            category_outputs = dict()
            # Segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            pd_gpu_place = fluid.CUDAPlace(dev_id)
            pd_cpu_place = fluid.CPUPlace()

            category_pd_type = dict()
            category_place = dict()
            category_tensors = dict()
            category_shapes = dict()
            category_lengths = dict()
            for cat, out in category_outputs.items():
                lod = self.normalized_map[cat]
                assert out.is_dense_tensor() or lod > 0, \
                    "non-dense tensor lists must have LoD > 0"

                if lod > 0:
                    # +1 for batch dim
                    seq_len = recursive_length(out, lod + 1)[1:]
                    shape = out.at(0).shape
                    if callable(shape):
                        shape = shape()
                    shape = [sum(seq_len[-1])] + list(shape[lod:])
                    category_shapes[cat] = shape
                    category_lengths[cat] = seq_len
                else:
                    out = out.as_tensor()
                    category_shapes[cat] = out.shape()
                    category_lengths[cat] = []

                category_tensors[cat] = out
                category_pd_type[cat] = to_paddle_type(out)
                if isinstance(out, (TensorGPU, TensorListGPU)):
                    category_place[cat] = pd_gpu_place
                else:
                    category_place[cat] = pd_cpu_place

            if self._data_batches[i] is None:
                pd_tensors = {}
                for cat, lod in self.normalized_map.items():
                    lod_tensor = fluid.core.LoDTensor()
                    lod_tensor._set_dims(category_shapes[cat])
                    pd_tensors[cat] = lod_tensor
                self._data_batches[i] = pd_tensors
            else:
                pd_tensors = self._data_batches[i]

            # Copy data from DALI Tensors to LoDTensors
            for cat, tensor in category_tensors.items():
                if hasattr(tensor, 'shape'):  # could be tensor list
                    assert self._dynamic_shape or \
                        tensor.shape() == pd_tensors[cat].shape(), \
                        ("Shapes do not match: DALI tensor has size {0}, "
                         "but LoDTensor has size {1}".format(
                             tensor.shape(), pd_tensors[cat].shape()))

                lod_tensor = pd_tensors[cat]
                lod_tensor._set_dims(category_shapes[cat])
                seq_len = category_lengths[cat]
                lod_tensor.set_recursive_sequence_lengths(seq_len)
                ptr = lod_tensor._mutable_data(category_place[cat],
                                               category_pd_type[cat])
                feed_ndarray(tensor, ptr)

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.release_outputs()
                p.schedule_run()

        self._counter += self._num_gpus * self.batch_size

        if (not self._fill_last_batch) and (self._counter > self._size) and self._size > 0:
            # First calculate how much data is required to
            # return exactly self._size entries.
            diff = self._num_gpus * self.batch_size - (self._counter
                                                       - self._size)
            # Figure out how many GPUs to grab from.
            num_gpus_to_grab = int(math.ceil(diff / self.batch_size))
            # Figure out how many results to grab from the last GPU
            # (as a fractional GPU batch may be required to bring us
            # right up to self._size).
            mod_diff = diff % self.batch_size
            data_from_last_gpu = mod_diff if mod_diff else self.batch_size

            # Grab the relevant data.
            # 1) Grab everything from the relevant GPUs.
            # 2) Grab the right data from the last GPU.
            # 3) Append data together correctly and return.
            output = self._data_batches[0:num_gpus_to_grab]
            output[-1] = output[-1].copy()
            for cat in self.output_map:
                lod_tensor = output[-1][cat]
                output[-1][cat] = lod_tensor_clip(lod_tensor,
                                                  data_from_last_gpu)
            return output

        return self._data_batches

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter >= self._size or self._size < 0:
            if self._fill_last_batch and not self._last_batch_padded:
                self._counter = self._counter % self._size
            else:
               self._counter = 0
            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")


class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for Paddle. It returns 2 outputs
    (data and label) in the form of LoDTensor.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to return a fraction of a full batch of data
                 such that the total entries returned by the
                 iterator == 'size'. Setting this flag to False will
                 cause the iterator to return the first integer multiple
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the LoDtensor will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one.

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    fill_last_batch = False, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    fill_last_batch = False, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    fill_last_batch = True, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    fill_last_batch = True, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 size,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        super(DALIClassificationIterator, self).__init__(
            pipelines, ["data", "label"], size, auto_reset=auto_reset,
            fill_last_batch=fill_last_batch,
            dynamic_shape=dynamic_shape,
            last_batch_padded=last_batch_padded)
