# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import math

import numpy as np
import paddle
from packaging.version import Version
import paddle.utils

from nvidia.dali import types
from nvidia.dali.backend import TensorListCPU, TensorGPU, TensorListGPU
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

if isinstance(paddle.__version__, str):
    assert Version(paddle.__version__) == Version("0.0.0") or Version(
        paddle.__version__
    ) >= Version("2.0.0"), "DALI PaddlePaddle support requires Paddle develop or release >= 2.0.0"


dtype_map = {
    types.DALIDataType.BOOL: paddle.framework.core.VarDesc.VarType.BOOL,
    types.DALIDataType.FLOAT: paddle.framework.core.VarDesc.VarType.FP32,
    types.DALIDataType.FLOAT64: paddle.framework.core.VarDesc.VarType.FP64,
    types.DALIDataType.FLOAT16: paddle.framework.core.VarDesc.VarType.FP16,
    types.DALIDataType.UINT8: paddle.framework.core.VarDesc.VarType.UINT8,
    types.DALIDataType.INT8: paddle.framework.core.VarDesc.VarType.INT8,
    types.DALIDataType.INT16: paddle.framework.core.VarDesc.VarType.INT16,
    types.DALIDataType.INT32: paddle.framework.core.VarDesc.VarType.INT32,
    types.DALIDataType.INT64: paddle.framework.core.VarDesc.VarType.INT64,
}


def to_paddle_type(tensor):
    r"""
    Get paddle dtype for given tensor or tensor list

    Args:
        tensor: tensor or tensor list

    Returns: paddle.framework.core.VarDesc.VarType
    """
    return dtype_map[tensor.dtype]


def feed_ndarray(dali_tensor, ptr, cuda_stream=None):
    """
    Copy contents of DALI tensor to Paddle's Tensor.

    Parameters
    ----------
    dali_tensor : dali.backend.TensorCPU or dali.backend.TensorGPU
                    Tensor from which to copy
    ptr : LoDTensor data pointer
            Destination of the copy
    cuda_stream : cudaStream_t handle or any value that can be cast to cudaStream_t
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
    """

    non_blocking = cuda_stream is not None
    cuda_stream = types._raw_cuda_stream_ptr(cuda_stream)

    c_type_pointer = ctypes.c_void_p(ptr)
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, cuda_stream, non_blocking=non_blocking)
    else:
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
            elif hasattr(data, "shape"):
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
    output = paddle.framework.core.LoDTensor()
    ndarray = np.array(lod_tensor)
    seq_len = lod_tensor.recursive_sequence_lengths()
    if not seq_len:
        output.set(ndarray[0:size], paddle.CPUPlace())
    else:
        last_len = size
        out_seq_len = []
        for lengths in seq_len:
            lengths = lengths[0:last_len]
            out_seq_len.append(lengths)
            last_len = sum(lengths)
        output.set(ndarray[0 : sum(out_seq_len[-1])], paddle.CPUPlace())
        output.set_recursive_sequence_lengths(out_seq_len)
    return output


class DALIGenericIterator(_DaliBaseIterator):
    """
    General DALI iterator for Paddle. It can return any number of
    outputs from the DALI pipeline in the form of Paddle's Tensors.

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    output_map : list of str or pair of type (str, int)
                The strings maps consecutive outputs of DALI pipelines to
                user specified name. Outputs will be returned from iterator
                as dictionary of those names. Each name should be distinct.
                Item can also be a pair of (str, int), where the int value
                specifies the LoD level of the resulting LoDTensor.
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried for the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets
                `last_batch_padded` accordingly to match the reader's configuration.
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use `last_batch_policy` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with `last_batch_policy` it tells
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

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
     next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(
        self,
        pipelines,
        output_map,
        size=-1,
        reader_name=None,
        auto_reset=False,
        fill_last_batch=None,
        dynamic_shape=False,
        last_batch_padded=False,
        last_batch_policy=LastBatchPolicy.FILL,
        prepare_first_batch=True,
    ):
        normalized_map = {}
        for v in output_map:
            if isinstance(v, str):
                normalized_map[v] = 0
            else:
                normalized_map[v[0]] = v[1]
        self.normalized_map = normalized_map

        # check the assert first as _DaliBaseIterator would run the prefetch
        output_map = [isinstance(v, str) and v or v[0] for v in output_map]
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self.output_map = output_map

        _DaliBaseIterator.__init__(
            self,
            pipelines,
            size,
            reader_name,
            auto_reset,
            fill_last_batch,
            last_batch_padded,
            last_batch_policy,
            prepare_first_batch=prepare_first_batch,
        )

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                self._report_no_data_in_pipeline()

    def __next__(self):
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]

        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            copy = not self._pipes[i].exec_dynamic
            # Initialize dict for all output categories
            category_outputs = dict()
            # Segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            pd_gpu_place = paddle.CUDAPlace(dev_id)
            pd_cpu_place = paddle.CPUPlace()

            category_pd_type = dict()
            category_place = dict()
            category_tensors = dict()
            category_shapes = dict()
            category_lengths = dict()
            for cat, out in category_outputs.items():
                lod = self.normalized_map[cat]
                assert out.is_dense_tensor() or lod > 0, "non-dense tensor lists must have LoD > 0"

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

            pd_tensors = {}
            data_batches[i] = pd_tensors
            stream = paddle.device.cuda.current_stream(dev_id).cuda_stream
            if copy:
                for cat, tensor in category_tensors.items():
                    lod_tensor = paddle.framework.core.LoDTensor()
                    pd_tensors[cat] = lod_tensor
                    lod_tensor._set_dims(category_shapes[cat])
                    seq_len = category_lengths[cat]
                    lod_tensor.set_recursive_sequence_lengths(seq_len)
                    lod_tensor._mutable_data(category_place[cat], category_pd_type[cat])

                for cat, tensor in category_tensors.items():
                    ptr = pd_tensors[cat]._mutable_data(category_place[cat], category_pd_type[cat])
                    feed_ndarray(tensor, ptr, stream)
            else:
                for cat, tensor in category_tensors.items():
                    capsule = tensor.__dlpack__(stream=stream)
                    pd_tensor = paddle.framework.core.from_dlpack(capsule)
                    seq_len = category_lengths[cat]
                    pd_tensor.set_recursive_sequence_lengths(seq_len)
                    pd_tensors[cat] = pd_tensor

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for cat in self.output_map:
                        batch[cat] = lod_tensor_clip(batch[cat], to_copy)
                    output.append(batch)
                return output

        else:
            if (
                self._last_batch_policy == LastBatchPolicy.PARTIAL
                and (self._counter > self._size)
                and self._size > 0
            ):
                # First calculate how much data is required to
                # return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
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
                output = data_batches[0:num_gpus_to_grab]
                output[-1] = output[-1].copy()
                for cat in self.output_map:
                    lod_tensor = output[-1][cat]
                    output[-1][cat] = lod_tensor_clip(lod_tensor, data_from_last_gpu)
                return output

        return data_batches


class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for Paddle. It returns 2 outputs
    (data and label) in the form of LoDTensor.

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
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried for the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets
                `last_batch_padded` accordingly to match the reader's configuration.
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use `last_batch_policy` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with `last_batch_policy` it tells
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

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(
        self,
        pipelines,
        size=-1,
        reader_name=None,
        auto_reset=False,
        fill_last_batch=None,
        dynamic_shape=False,
        last_batch_padded=False,
        last_batch_policy=LastBatchPolicy.FILL,
        prepare_first_batch=True,
    ):
        super(DALIClassificationIterator, self).__init__(
            pipelines,
            ["data", "label"],
            size,
            reader_name=reader_name,
            auto_reset=auto_reset,
            fill_last_batch=fill_last_batch,
            dynamic_shape=dynamic_shape,
            last_batch_padded=last_batch_padded,
            last_batch_policy=last_batch_policy,
            prepare_first_batch=prepare_first_batch,
        )
