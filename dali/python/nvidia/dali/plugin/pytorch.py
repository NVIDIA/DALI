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
from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import logging

import numpy as np

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}

def feed_ndarray(dali_tensor, arr, cuda_stream = None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream or any value that can be casted to cudaStream_t
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, cuda_stream)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr

class DALIGenericIterator(object):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Please keep in mind that Tensors returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    output_map : list of str
                 List of strings which maps consecutive outputs
                 of DALI pipelines to user specified name.
                 Outputs will be returned from iterator as dictionary
                 of those names.
                 Each name should be distinct
    size : int, default = -1
           Number of samples in the epoch (Usually the size of the dataset).
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
           Name of the reader which will be queried to the shard size, number of shards and
           all other properties necessary to count properly the number of relevant and padded
           samples that iterator needs to deal with. It automatically sets `fill_last_batch` and
           `last_batch_padded` accordingly
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one. For this to happen, the option ``pad_last_batch``
                 in the reader needs to be set to ``True`` as well.
                 It is overwritten when `reader_name` argument is provided

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
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = int(size)
        self._auto_reset = auto_reset
        self._dynamic_shape = dynamic_shape
        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and (len(pipelines) == 1 or reader_name)), "Negative size is supported only for a single pipeline"
        assert not reader_name or (reader_name and self._size < 0), "When reader_name is provided, size should not be set"
        if self._size < 0 and not reader_name:
            self._auto_reset = False
            self._fill_last_batch = False
            self._last_batch_padded = False
        self._pipes = pipelines
        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()

        self._reader_name = reader_name
        if self._reader_name:
            self._size_no_pad = self._pipes[0].epoch_size(self._reader_name, False)
            assert np.all(np.equal([p.epoch_size(self._reader_name, False) for p in self._pipes], self._size_no_pad)), \
                "All pipelines readers should have the same size, check if they are reading the same data"

            self._shards_num = self._pipes[0].shards_number(self._reader_name)
            assert np.all(np.equal([p.shards_number(self._reader_name) for p in self._pipes], self._shards_num)), \
                "All pipelines readers should have the same shard number set"

            self._shards_id = [p.shard_id(self._reader_name) for p in self._pipes]

            assert np.all([p.if_reader_pads(self._reader_name) for p in self._pipes]) or \
                   not np.any([p.if_reader_pads(self._reader_name) for p in self._pipes]), \
                "All pipelines readers should have set padding in the same way"
            self._last_batch_padded = self._pipes[0].if_reader_pads(self._reader_name)

            assert np.all([p.if_sticks_to_shard(self._reader_name) for p in self._pipes]) or \
                   not np.any([p.if_sticks_to_shard(self._reader_name) for p in self._pipes]), \
                "All pipelines readers should have set stick to the shard in the same way"
            self._if_sticks_to_shard = self._pipes[0].if_sticks_to_shard(self._reader_name)

            self._size = self._pipes[0].epoch_size(self._reader_name, True) // self._shards_num

        # Use double-buffering of data batches
        self._data_batches = [None for i in range(self._num_gpus)]
        self._counter = 0
        self._counter_per_gpu = [0] * self._num_gpus
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
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
            # initialize dict for all output categories
            category_outputs = dict()
            # segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_shapes = dict()
            for category, out in category_outputs.items():
                category_tensors[category] = out.as_tensor()
                category_shapes[category] = category_tensors[category].shape()

            # If we did not yet allocate memory for that batch, do it now
            if self._data_batches[i] is None:
                category_torch_type = dict()
                category_device = dict()
                torch_gpu_device = torch.device('cuda', dev_id)
                torch_cpu_device = torch.device('cpu')
                # check category and device
                for category in self._output_categories:
                    category_torch_type[category] = to_torch_type[np.dtype(category_tensors[category].dtype())]
                    if type(category_tensors[category]) is TensorGPU:
                        category_device[category] = torch_gpu_device
                    else:
                        category_device[category] = torch_cpu_device

                pyt_tensors = dict()
                for category in self._output_categories:
                    pyt_tensors[category] = torch.zeros(category_shapes[category],
                                                        dtype=category_torch_type[category],
                                                        device=category_device[category])

                self._data_batches[i] = pyt_tensors
            else:
                pyt_tensors = self._data_batches[i]

            # Copy data from DALI Tensors to torch tensors
            for category, tensor in category_tensors.items():
                if self._dynamic_shape and tensor.shape() != list(pyt_tensors[category].size()):
                    pyt_tensors[category] = torch.zeros(category_shapes[category],
                                                        dtype=pyt_tensors[category].dtype,
                                                        device=pyt_tensors[category].device)
                if isinstance(tensor, (TensorGPU, TensorListGPU)):
                    # Using same cuda_stream used by torch.zeros to set the memory
                    stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                    feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                else:
                    feed_ndarray(tensor, pyt_tensors[category])

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.release_outputs()
                p.schedule_run()

        if self._reader_name:
            self._counter += self.batch_size
            if not self._fill_last_batch:
                # calculate each shard size for each id, and check how many samples are left by substracting from iterator counter
                # the shard size, then go though all GPUs and return only relevant data by stripping padded one
                left = [self._counter - int((id + 1) * self._size_no_pad / self._shards_num) + int(id * self._size_no_pad / self._shards_num) for id in self._shards_id]
                left = [self.batch_size - l for l in left]
                if_drop = np.less(left, self.batch_size)
                if np.any(if_drop):
                    output = []
                    for batch, to_copy in zip(self._data_batches, left):
                        batch = batch.copy()
                        for category in self._output_categories:
                            batch[category] = batch[category][0:to_copy]
                        output.append(batch)
                    return output

        else:
            self._counter += self._num_gpus * self.batch_size
            if (not self._fill_last_batch) and (self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff/self.batch_size))
                # Figure out how many results to grab from the last GPU (as a fractional GPU batch may be required to
                # bring us right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = self._data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
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
            # advance to the next shard
            if self._reader_name and not self._if_sticks_to_shard:
                self._shards_id = [(v + 1) % self._shards_num for v in self._shards_id]
            if self._fill_last_batch and not self._last_batch_padded:
                if self._reader_name:
                    self._counter_per_gpu = [v - int((id + 1) * self._size_no_pad / self._shards_num) + int(id * self._size_no_pad / self._shards_num) for v, id in zip(self._counter_per_gpu, self._shards_id)]
                    self._counter = min(self._counter_per_gpu)
                else:
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
    DALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

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
    size : int, default = -1
           Number of samples in the epoch (Usually the size of the dataset).
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
           Name of the reader which will be queried to the shard size, number of shards and
           all other properties necessary to count properly the number of relevant and padded
           samples that iterator needs to deal with. It automatically sets `fill_last_batch` and
           `last_batch_padded` accordingly
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one. For this to happen, the option ``pad_last_batch``
                 in the reader needs to be set to ``True`` as well.
                 It is overwritten when `reader_name` argument is provided

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
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        super(DALIClassificationIterator, self).__init__(pipelines, ["data", "label"],
                                                         size, reader_name=reader_name,
                                                         auto_reset = auto_reset,
                                                         fill_last_batch = fill_last_batch,
                                                         dynamic_shape = dynamic_shape,
                                                         last_batch_padded = last_batch_padded)


class TorchPythonFunction(ops.PythonFunctionBase):
    ops.register_cpu_op('TorchPythonFunction')
    ops.register_gpu_op('TorchPythonFunction')

    def _torch_stream_wrapper(self, function, *ins):
        with torch.cuda.stream(self.stream):
            out = function(*ins)
        self.stream.synchronize()
        return out

    def torch_wrapper(self, batch_processing, function, device, *args):
        func = function if device == 'cpu' else \
               lambda *ins: self._torch_stream_wrapper(function, *ins)
        if batch_processing:
            return ops.PythonFunction.function_wrapper_batch(func,
                                                             torch.utils.dlpack.from_dlpack,
                                                             torch.utils.dlpack.to_dlpack,
                                                             *args)
        else:
            return ops.PythonFunction.function_wrapper_per_sample(func,
                                                                  torch_dlpack.from_dlpack,
                                                                  torch_dlpack.to_dlpack,
                                                                  *args)

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if pipeline is None:
            Pipeline._raise_no_current_pipeline("TorchPythonFunction")
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=pipeline.device_id)
        return super(TorchPythonFunction, self).__call__(*inputs, **kwargs)

    def __init__(self, function, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        self.stream = None
        super(TorchPythonFunction, self).__init__(impl_name="DLTensorPythonFunctionImpl",
                                                  function=lambda *ins:
                                                  self.torch_wrapper(batch_processing,
                                                                    function, device,
                                                                    *ins),
                                                  num_outputs=num_outputs, device=device,
                                                  batch_processing=batch_processing, **kwargs)
