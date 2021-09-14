# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.backend import TensorGPU, TensorListGPU, TensorListCPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import math
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
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    if isinstance(dali_tensor, (TensorListCPU, TensorListGPU)):
        dali_type = dali_tensor[0].dtype()
    else:
        dali_type = dali_tensor.dtype()
    dali_type = np.dtype(dali_type)

    assert to_torch_type[dali_type] == arr.dtype, ("The element type of DALI Tensor/TensorList"
            " doesn't match the element type of the target PyTorch Tensor: {} vs {}".format(to_torch_type[dali_type], arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, None if cuda_stream is None else ctypes.c_void_p(cuda_stream))
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr

class DALIGenericIterator(_DaliBaseIterator):
    """
    General DALI iterator for PyTorch. It can return any number of
    outputs from the DALI pipeline in the form of PyTorch's Tensors.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str
                 List of strings which maps consecutive outputs
                 of DALI pipelines to user specified name.
                 Outputs will be returned from iterator as dictionary
                 of those names.
                 Each name should be distinct
    size : int, default = -1
           Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):

        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        _DaliBaseIterator.__init__(self,
                                   pipelines,
                                   size,
                                   reader_name,
                                   auto_reset,
                                   fill_last_batch,
                                   last_batch_padded,
                                   last_batch_policy,
                                   prepare_first_batch=prepare_first_batch)

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline. This may happen if `last_batch_policy` is set to PARTIAL and the requested batch size is greater than the shard size."

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]
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

            category_torch_type = dict()
            category_device = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device('cpu')
            # check category and device
            for category in self._output_categories:
                category_torch_type[category] = to_torch_type[np.dtype(category_tensors[category].dtype())]
                if type(category_tensors[category]) is TensorGPU:
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device('cuda', dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            pyt_tensors = dict()
            for category in self._output_categories:
                pyt_tensors[category] = torch.empty(category_shapes[category],
                                                    dtype=category_torch_type[category],
                                                    device=category_device[category])

            data_batches[i] = pyt_tensors

            # Copy data from DALI Tensors to torch tensors
            for category, tensor in category_tensors.items():
                if isinstance(tensor, (TensorGPU, TensorListGPU)):
                    # Using same cuda_stream used by torch.zeros to set the memory
                    stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                    feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                else:
                    feed_ndarray(tensor, pyt_tensors[category])

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for category in self._output_categories:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (self._counter > self._size) and self._size > 0:
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
                output = data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

        return data_batches

class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, reader_name)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines, ["data", "label"], reader_name)

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    size : int, default = -1
           Number of samples in the shard for the wrapped pipeline (if there is more than one it is a sum)
           Providing -1 means that the iterator will work until StopIteration is raised
           from the inside of iter_setup(). The options `last_batch_policy`, `last_batch_padded` and
           `auto_reset` don't work in such case. It works with only one pipeline inside
           the iterator.
           Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy` to
                PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy : default = FILL
                What to do with the last batch when there is no enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = PARTIAL, last_batch_padded = True  -> last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = PARTIAL, last_batch_padded = False -> last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = FILL, last_batch_padded = True   -> last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = FILL, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = DROP, last_batch_padded = True   -> last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = DROP, last_batch_padded = False  -> last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):
        super(DALIClassificationIterator, self).__init__(pipelines, ["data", "label"],
                                                         size,
                                                         reader_name=reader_name,
                                                         auto_reset=auto_reset,
                                                         fill_last_batch=fill_last_batch,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded=last_batch_padded,
                                                         last_batch_policy=last_batch_policy,
                                                         prepare_first_batch=prepare_first_batch)


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
                                                             self.num_outputs,
                                                             torch.utils.dlpack.from_dlpack,
                                                             torch.utils.dlpack.to_dlpack,
                                                             *args)
        else:
            return ops.PythonFunction.function_wrapper_per_sample(func,
                                                                  self.num_outputs,
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



ops._wrap_op(TorchPythonFunction, "fn", __name__)
