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

import sys

from typing import Union, Optional
from typing import Any, Dict, List

from nvidia.dali import internal as _internal
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.backend import TensorCPU, TensorGPU, TensorListCPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline

from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

import torch
import torch.multiprocessing as mp
import torch.utils.dlpack as torch_dlpack  # noqa: F401
from torch.utils import data
from torch.utils.data._utils.collate import collate
from torch.utils.data.dataloader import (
    DataLoader,
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
    _BaseDataLoaderIter,
)
import ctypes
import numpy as np
import threading
from queue import Empty

from nvtx import nvtx

from . import fn  # noqa: F401

from nvidia.dali.plugin.pytorch._torch_function import TorchPythonFunction as TorchPythonFunction

_internal._adjust_operator_module(TorchPythonFunction, sys.modules[__name__], [])

ops._wrap_op(TorchPythonFunction, "fn", __name__)


to_torch_type = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.BOOL: torch.bool,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64,
}


def to_torch_tensor(tensor_or_tl, device_id=0):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `tensor_or_tl` : TensorGPU or TensorListGPU
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    if isinstance(dali_tensor, (TensorGPU)):
        torch_device = torch.device("cuda", device_id)
    else:
        torch_device = torch.device("cpu")

    out_torch = torch.empty(
        dali_tensor.shape(),
        dtype=to_torch_type[dali_tensor.dtype],
        device=torch_device,
    )

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(out_torch.data_ptr())
    if isinstance(dali_tensor, (TensorGPU)):
        non_blocking = True
        cuda_stream = torch.cuda.current_stream(device=torch_device)
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor_or_tl.copy_to_external(c_type_pointer, stream, non_blocking)
    else:
        tensor_or_tl.copy_to_external(c_type_pointer)

    return out_torch


def feed_ndarray(
    dali_tensor: Union[TensorCPU, TensorGPU, TensorListCPU, TensorListGPU],
    arr: torch.Tensor,
    cuda_stream: Union[torch.cuda.Stream, Any, None] = None,
) -> torch.Tensor:
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    dali_tensor : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    arr : torch.Tensor
            Destination of the copy
    cuda_stream : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = to_torch_type[dali_tensor.dtype]

    assert dali_type == arr.dtype, (
        "The element type of DALI Tensor/TensorList"
        " doesn't match the element type of the target PyTorch Tensor: "
        "{} vs {}".format(dali_type, arr.dtype)
    )
    assert dali_tensor.shape() == list(
        arr.size()
    ), "Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".format(
        dali_tensor.shape(), list(arr.size())
    )

    non_blocking = cuda_stream is not None
    cuda_stream = types._raw_cuda_stream_ptr(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        dali_tensor.copy_to_external(c_type_pointer, cuda_stream, non_blocking=non_blocking)
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
        pipelines: Union[List[Pipeline], Pipeline],
        output_map: List[str],
        size: int = -1,
        reader_name: Optional[str] = None,
        auto_reset: Union[str, bool, None] = False,
        fill_last_batch: Optional[bool] = None,
        dynamic_shape: Optional[bool] = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
        prepare_first_batch: bool = True,
    ) -> None:
        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
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

    def __next__(self) -> List[Dict[str, torch.Tensor]]:
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
            is_exec_dynamic = self._pipes[i].exec_dynamic
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
            torch_cpu_device = torch.device("cpu")
            # check category and device
            for category in self._output_categories:
                category_torch_type[category] = to_torch_type[category_tensors[category].dtype]
                if type(category_tensors[category]) is TensorGPU:
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device("cuda", dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            pyt_tensors = dict()

            copy = not is_exec_dynamic
            if copy:
                # Copy data from DALI Tensors to torch tensors
                for category, tensor in category_tensors.items():
                    pyt_tensor = torch.empty(
                        category_shapes[category],
                        dtype=category_torch_type[category],
                        device=category_device[category],
                    )
                    pyt_tensors[category] = pyt_tensor

                    if isinstance(tensor, TensorGPU):
                        # Using same cuda_stream used by torch.zeros to set the memory
                        stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                        feed_ndarray(tensor, pyt_tensor, cuda_stream=stream)
                    elif isinstance(tensor, TensorCPU):
                        feed_ndarray(tensor, pyt_tensor)
                    else:
                        raise RuntimeError(
                            f"Internal error: unexpected type {type(tensor)}.\n"
                            f"Expected TensorCPU or TensorGPU"
                        )
            else:
                for category, tensor in category_tensors.items():
                    with category_device[category]:
                        pyt_tensor = torch.from_dlpack(tensor)
                        pyt_tensors[category] = pyt_tensor

            data_batches[i] = pyt_tensors

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
            if (
                self._last_batch_policy == LastBatchPolicy.PARTIAL
                and (self._counter > self._size)
                and self._size > 0
            ):
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
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
        pipelines: Union[List[Pipeline], Pipeline],
        size: int = -1,
        reader_name: Optional[str] = None,
        auto_reset: Union[str, bool, None] = False,
        fill_last_batch: Optional[bool] = None,
        dynamic_shape: Optional[bool] = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
        prepare_first_batch: bool = True,
    ) -> None:
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


class DALIRaggedIterator(_DaliBaseIterator):
    """
    General DALI iterator for PyTorch with ragged tensors.
    It can return any number of outputs from the DALI pipeline
    in the form of per GPU dictionaries.
    These dictionaries consisting of PyTorch Tensors
    (for outputs marked as DALIRaggedIterator.DENSE_TAG),
    sparse COO PyTorch Tensors
    (for outputs marked as DALIRaggedIterator.SPARSE_COO_TAG)
    and list of PyTorch Tensors
    (for outputs marked as DALIRaggedIterator.SPARSE_LIST_TAG).

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
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    output_types : list of str, optional, default = None
                List of tags indicating whether the pipeline(s) output batch is
                uniform (all the samples have the same size) or not. Batch output marked
                as the former will be returned as a single PyTorch Tensor, the latter
                will be returned as a specified sparse PyTorch Tensor format.
                Must be either DALIRaggedIterator.DENSE_TAG
                or DALIRaggedIterator.SPARSE_LIST_TAG
                or DALIRaggedIterator.SPARSE_COO_TAG
                Length of output_types must match the number of output of the pipeline(s).
                If not set, all outputs are considered to be marked with
                DALIRaggedIterator.DENSE_TAG.
                For now sparse mode supports only list of tensors and COO sparse tensor format.
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

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
                the next epoch (it doesn't literally drop but sets ``pad`` field of ndarray
                so the following code could use it to drop the data). If set to ``False`` next
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

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  ->
    last batch = ``[7]``, next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False ->
    last batch = ``[7]``, next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   ->
    last batch = ``[7, 7]``, next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  ->
    last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   ->
    last batch = ``[5, 6]``, next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  ->
    last batch = ``[5, 6]``, next iteration will return ``[2, 3]``
    """

    def __init__(
        self,
        pipelines: Union[List[Pipeline], Pipeline],
        output_map: List[str],
        size: int = -1,
        reader_name: Optional[str] = None,
        output_types: Optional[List[str]] = None,
        auto_reset: Union[str, bool, None] = False,
        fill_last_batch: Optional[bool] = None,
        dynamic_shape: Optional[bool] = False,
        last_batch_padded: bool = False,
        last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
        prepare_first_batch: bool = True,
    ) -> None:
        # check the assert first as _DaliBaseIterator would run the prefetch
        self._output_tags = {
            DALIRaggedIterator.DENSE_TAG,
            DALIRaggedIterator.SPARSE_LIST_TAG,
            DALIRaggedIterator.SPARSE_COO_TAG,
        }

        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        assert (
            output_types is None or set(output_types) <= self._output_tags
        ), "Only DENSE_TAG, SPARSE_LIST_TAG and SPARSE_COO_TAG are allowed"

        self.output_map = output_map
        self._outputs_types = output_types

        super(DALIRaggedIterator, self).__init__(
            pipelines,
            size,
            reader_name,
            auto_reset,
            fill_last_batch,
            last_batch_padded,
            last_batch_policy,
            prepare_first_batch,
        )

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = self._first_batch = DALIRaggedIterator.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                self._report_no_data_in_pipeline()

    def __next__(self) -> List[Dict[str, torch.Tensor]]:
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        dali_outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            is_exec_dynamic = self._pipes[i].exec_dynamic
            # initialize dict for all output categories
            category_outputs = dict()
            # segregate outputs into categories
            for j, out in enumerate(dali_outputs[i]):
                category_outputs[self.output_map[j]] = out

            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_shapes = dict()
            category_torch_type = dict()
            category_device = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device("cpu")

            for j, (category, out) in enumerate(category_outputs.items()):
                if (
                    self._outputs_types is None
                    or self._outputs_types[j] == DALIRaggedIterator.DENSE_TAG
                ):
                    category_tensors[category] = out.as_tensor()
                    category_shapes[category] = category_tensors[category].shape()
                else:
                    category_tensors[category] = [x for x in out]
                    category_shapes[category] = [x.shape() for x in out]

                # check dtype
                category_torch_type[category] = to_torch_type[out.dtype]

                # check device
                if type(out) is TensorListGPU:
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device("cuda", dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            copy = not is_exec_dynamic

            pyt_tensors = dict()
            if copy:
                for j, category in enumerate(self.output_map):
                    if (
                        self._outputs_types is None
                        or self._outputs_types[j] == DALIRaggedIterator.DENSE_TAG
                    ):
                        pyt_tensors[category] = torch.empty(
                            category_shapes[category],
                            dtype=category_torch_type[category],
                            device=category_device[category],
                        )
                    else:
                        pyt_tensors[category] = [
                            torch.empty(
                                shape,
                                dtype=category_torch_type[category],
                                device=category_device[category],
                            )
                            for shape in category_shapes[category]
                        ]

                # Copy data from DALI Tensors to torch tensors
                for j, (category, tensor) in enumerate(category_tensors.items()):
                    if (
                        self._outputs_types is None
                        or self._outputs_types[j] == DALIRaggedIterator.DENSE_TAG
                    ):
                        if isinstance(tensor, (TensorGPU, TensorListGPU)):
                            # Using same cuda_stream used by torch.zeros to set the memory
                            stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                            feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                        else:
                            feed_ndarray(tensor, pyt_tensors[category])
                    else:
                        for k, single_tensor in enumerate(tensor):
                            if isinstance(tensor, (TensorGPU, TensorListGPU)):
                                # Using same cuda_stream used by torch.zeros to set the memory
                                stream = torch.cuda.current_stream(
                                    device=pyt_tensors[category][k].device
                                )
                                feed_ndarray(
                                    single_tensor, pyt_tensors[category][k], cuda_stream=stream
                                )
                            else:
                                feed_ndarray(single_tensor, pyt_tensors[category][k])

                        if self._outputs_types[j] == DALIRaggedIterator.SPARSE_COO_TAG:
                            values = torch.hstack(pyt_tensors[category])

                            indices = [
                                [(i, j) for j in range(shape[0])]
                                for i, shape in enumerate(category_shapes[category])
                            ]
                            indices = [index for el_indices in indices for index in el_indices]
                            indices = torch.LongTensor(indices, device=values.device)

                            pyt_tensors[category] = torch.sparse_coo_tensor(indices.T, values)
            else:
                for j, (category, tensor_or_tl) in enumerate(category_tensors.items()):
                    with category_device[category]:
                        if isinstance(tensor_or_tl, list):
                            pyt_tl = [torch.from_dlpack(t) for t in tensor_or_tl]
                            if self._outputs_types[j] == DALIRaggedIterator.SPARSE_COO_TAG:
                                values = torch.hstack(pyt_tl)
                                indices = [
                                    [(i, j) for j in range(shape[0])]
                                    for i, shape in enumerate(category_shapes[category])
                                ]
                                indices = [index for el_indices in indices for index in el_indices]
                                indices = torch.LongTensor(indices, device=values.device)

                                pyt_tl = torch.sparse_coo_tensor(indices.T, values)

                            pyt_tensors[category] = pyt_tl
                        else:
                            pyt_tensors[category] = torch.from_dlpack(tensor_or_tl)

            data_batches[i] = pyt_tensors

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for category in self.output_map:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if (
                self._last_batch_policy == LastBatchPolicy.PARTIAL
                and (self._counter > self._size)
                and self._size > 0
            ):
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
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

    DENSE_TAG: str = "dense"
    SPARSE_LIST_TAG: str = "sparse_list"
    SPARSE_COO_TAG: str = "sparse_coo"


class DALIProxy:
    """
    Proxy to communicate to send processing requests to a DALI pipeline running on the main loop.
    This is used by PyTorch data workers to assign some processing to the loaded samples, which can
    execute in the main process via the GPU.

    Background: As the PyTorch workers run on separate processes, using the GPU directly from each
    of those is not ideal for performance, due to the usage of several CUDA contexts.
    """

    def __init__(self, input_names):
        """
        Initializes a new DALI proxy instance.

        Args:
            input_names (list): list of strings representing the inputs to the pipeline. Those should match
                                the names of the ``external_source`` nodes in the DALI pipeline.
        """
        self.input_names = input_names
        self.num_inputs = len(input_names)
        # Multi-process queue used to transfer data from the pytorch workers to the main process
        self.send_q = mp.Queue()
        # Multi-process queue used by the main process to remember the actual order of execution of the requests
        self.order_q = mp.Queue()
        # Torch worker id, to be filled on first call to worker_id()
        self._worker_id = None
        # Iteration index for the current worker
        self.data_idx = 0

    def worker_id(self):
        if self._worker_id is None:
            self._worker_id = torch.utils.data.get_worker_info().id
        return self._worker_id

    class PipelineOutputRef:
        """
        Placeholder for a pipeline output reference, after the iteration has been scheduled to DALI.
        """
        def __init__(self, info):
            self.info = info

    def schedule_batch(self, inputs):
        """
        Schedule a pipeline run to DALI, by queuing the worker id, the iteration index and the inputs
        """
        # Identifier of this request
        info = (self.worker_id(), self.data_idx)
        with nvtx.annotate(f"dali_proxy.send_q.put {info}", color="blue"):
            self.send_q.put((info, inputs))
        self.data_idx = self.data_idx + 1
        # Returns a placeholder, which is replaced with the actual data once the iteration completes
        return DALIProxy.PipelineOutputRef(info)

    class PipelineRunRef:
        """
        Placeholder for a pipeline run reference, which is returned by the data worker instead of the actual data

        The PyTorch worker returns this trivial object, only containing information about this proxy instance and
        the input data to the pipeline. Later in the collate function, we send the data for execution to DALI.
        """
        def __init__(self, dali_proxy, inputs):
            self.dali_proxy = dali_proxy
            self.inputs = inputs
            assert len(self.inputs) == dali_proxy.num_inputs

    def transform(self, *inputs):
        """
        The 'transform' function consists of returning a reference to the pipeline run
        """
        assert len(inputs) == self.num_inputs, f"Unexpected number of inputs: {len(inputs)}"
        return DALIProxy.PipelineRunRef(self, inputs)


def collate_pipeline_run_ref_fn(pipe_out, *, collate_fn_map=None):
    """
    Special collate function that schedules a batch for execution
    """
    assert len(pipe_out) > 0
    first_elem = pipe_out[0]
    inputs = [[] for idx in range(len(first_elem.inputs))]
    for elem in pipe_out:
        assert first_elem.dali_proxy == elem.dali_proxy
        for idx, input_ref in enumerate(elem.inputs):
            inputs[idx].append(input_ref)
    return first_elem.dali_proxy.schedule_batch(inputs)


def custom_collate(batch):
    """
    Subscribe a special collate function for PipelineRunRef, that handles the scheduling of the iteration
    on the fly
    """
    collate_fn_map = data._utils.collate.default_collate_fn_map
    collate_fn_map.update({DALIProxy.PipelineRunRef: collate_pipeline_run_ref_fn})
    return collate(batch, collate_fn_map=collate_fn_map)


def flatten_tuple(nested_tuple):
    """
    Flattens a nested tuple
    """
    flat_list = []

    def _flatten(t):
        for item in t:
            if isinstance(item, tuple):
                _flatten(item)
            else:
                flat_list.append(item)

    _flatten(nested_tuple)
    return tuple(flat_list)


class DALIMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    """
    Data loader iterator used by the DALI proxy data loader
    """
    def __init__(self, loader):
        super().__init__(loader)
        self.loader = loader

    def _next_data(self):
        data = super()._next_data()
        if not hasattr(data, "__iter__"):
            print(
                "Warning: Non iterable returned from dataloader. Please "
                " review the code, since it usually indicates a bug in a pipeline."
            )
            data = [data]
        for data_idx, data_elem in enumerate(data):
            # If loader returns a dictionary the iterator iterates over its keys.
            # We need to access a value. Probably need to address more casess.
            if isinstance(data, dict):
                if isinstance(data[data_elem], DALIProxy.PipelineOutputRef):
                    data[data_elem] = self.loader.get_outputs(data[data_elem].info)
            if isinstance(data_elem, DALIProxy.PipelineOutputRef):
                data[data_idx] = self.loader.get_outputs(data_elem.info)
        return data


class DALIDataLoader(DataLoader):
    """
    DALI data loader to be used in the main loop, which runs the DALI pipeline doing the processing
    asynchronously with regards to the training.
    """
    def __init__(self, pipe, dali_proxy, *args, **kwargs):
        if "collate_fn" in kwargs and kwargs["collate_fn"] is not None:
            print(
                "Warning: Make sure to handle DALIProxy.PipelineRunRef when providing"
                " a custom collate_fn"
            )
        else:
            kwargs["collate_fn"] = custom_collate
        super().__init__(*args, **kwargs)
        self.pipe = pipe
        self.dali_proxy = dali_proxy
        self.t = None
        self.thread_stop_event = None
        self.cache_outputs = dict()
        self.cache_inputs = dict()

    def outputs(self):
        # Get the information about the order of execution, so that we know which one is the next iteration
        torch.cuda.nvtx.range_push("order_q.get")
        info = self.dali_proxy.order_q.get()
        torch.cuda.nvtx.range_pop()

        # Get the outputs from the current iteration
        torch.cuda.nvtx.range_push(f"pipe.outputs {info}")
        outputs = self.pipe.outputs()
        torch.cuda.nvtx.range_pop()

        # Return information about the iteration, together with the data
        processed_outputs = tuple(
            [to_torch_tensor(output, device_id=self.pipe.device_id) for output in outputs]
        )
        return (info, processed_outputs)

    def get_outputs(self, req_info):
        req_outputs = None
        # If the data was already read, just return it (and clear the cache entry)
        if req_info in self.cache_outputs:
            req_outputs = self.cache_outputs[req_info]
            del self.cache_outputs[req_info]
            del self.cache_inputs[req_info]
        else:
            info = None
            # If not the data we are looking for, store it and keep processing until we find it
            while req_info != info:
                info, processed_outputs = self.outputs()
                if info == req_info:
                    req_outputs = processed_outputs
                    del self.cache_inputs[req_info]
                else:
                    self.cache_outputs[info] = processed_outputs
        # Unpack single element tuples
        if isinstance(req_outputs, tuple) and len(req_outputs) == 1:
            req_outputs = req_outputs[0]
        return req_outputs

    def thread_fn(self):
        """
        Asynchronous DALI thread that gets iteration data from the queue and schedules it for execution
        """
        while not self.thread_stop_event.is_set():
            try:
                torch.cuda.nvtx.range_push("dali_proxy.send_q.get")
                info, inputs = self.dali_proxy.send_q.get(timeout=5)
                torch.cuda.nvtx.range_pop()
                self.cache_inputs[info] = inputs
            except mp.TimeoutError:
                continue
            except Empty:
                continue
            torch.cuda.nvtx.range_push("dali_proxy.order_q.put {info}")
            self.dali_proxy.order_q.put(info)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"feed_input {info}")
            for idx, input_name in enumerate(self.dali_proxy.input_names):
                self.pipe.feed_input(input_name, inputs[idx])
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("schedule_run {info}")
            self.pipe.schedule_run()
            torch.cuda.nvtx.range_pop()

    def start_thread(self):
        """
        Starts the DALI pipeline thread
        """
        if self.t is not None:
            return
        self.t = threading.Thread(target=DALIDataLoader.thread_fn, args=(self,))
        self.thread_stop_event = threading.Event()
        self.t.start()

    def stop_thread(self):
        """
        Stops the DALI pipeline thread
        """
        if self.thread_stop_event is None:
            return
        self.thread_stop_event.set()
        self.t.join()
        self.t = None
        self.thread_stop_event = None

    def __enter__(self):
        self.start_thread()

    def __exit__(self, exc_type, exc_value, tb):
        self.stop_thread()

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return DALIMultiProcessingDataLoaderIter(self)
