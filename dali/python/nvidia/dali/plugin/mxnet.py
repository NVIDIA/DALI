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
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import types
import mxnet as mx
import ctypes
import logging
import numpy as np


##################################################
##################################################
################## Common utils ##################
##################################################
##################################################


# MXNet currently does not expose WaitToWrite C API call
# in Python API
def _wait_to_write(arr):
    if not isinstance(arr, mx.nd.NDArray):
        raise RuntimeError("Can only wait for NDArray")
    mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)

def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to MXNet's NDArray.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : mxnet.nd.NDArray
            Destination of the copy
    """
    # Wait until arr is no longer used by the engine
    _wait_to_write(arr)
    assert dali_tensor.shape() == list(arr.shape), \
            ("Shapes do not match: DALI tensor has shape {0}"
            ", but NDArray has shape {1}".format(dali_tensor.shape(), list(arr.shape)))
    # Get CTypes void pointer to the underlying memory held by arr
    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr.handle, ctypes.byref(ptr))
    # Copy data from DALI tensor to ptr
    dali_tensor.copy_to_external(ptr)

class _DALIIteratorBase(mx.io.DataIter):
    """
    Base class with methods shared by both DALIGenericIterator and DALIGluonIterator.
    """
    def __init__(self,
                 pipelines,
                 size,
                 fill_last_batch,
                 last_batch_padded,
                 auto_reset):
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._pipes = pipelines
        self._num_gpus = len(pipelines)
        self.batch_size = pipelines[0].batch_size
        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        self._counter = 0
        self._size = int(size)
        self._auto_reset = auto_reset
        assert self._size != 0, "Size cannot be 0"
        assert self._size > 0 or (self._size < 0 and len(pipelines) == 1), "Negative size is supported only for a single pipeline"
        if self._size < 0:
            self._auto_reset = False
            self._fill_last_batch = False
            self._last_batch_padded = False


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

    @property
    def size(self):
        return self._size

    def _check_iteration_stop(self):
        if self._counter >= self._size and self._size > 0:
            if self._auto_reset:
                self.reset()
            raise StopIteration

###################################################
###################################################
################## MXNet Sym API ##################
###################################################
###################################################

class DALIGenericIterator(_DALIIteratorBase):
    """
    General DALI iterator for MXNet. It can return any number of
    outputs from the DALI pipeline in the form of MXNet's DataBatch
    of NDArrays.

    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    output_map : list of (str, str)
                 List of pairs (output_name, tag) which maps consecutive
                 outputs of DALI pipelines to proper field in MXNet's
                 DataBatch.
                 tag is one of DALIGenericIterator.DATA_TAG
                 and DALIGenericIterator.LABEL_TAG mapping given output
                 for data or label correspondingly.
                 output_names should be distinct.
    size : int
          Number of samples in the epoch (Usually the size of the dataset).
          Providing -1 means that the iterator will work until StopIteration is raised
          from the inside of iter_setup(). The options `fill_last_batch`, `last_batch_padded` and
          `auto_reset` don't work in such case. It works with only one pipeline inside
          the iterator.
    data_layout : str, optional, default = 'NCHW'
                  Either 'NHWC' or 'NCHW' - layout of the pipeline outputs.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    squeeze_labels: bool, optional, default = True
                 Whether the iterator should squeeze the labels before
                 copying them to the ndarray.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the mxnet.ndarray will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
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
                 data_layout='NCHW',
                 fill_last_batch=True,
                 auto_reset=False,
                 squeeze_labels=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        super(DALIGenericIterator, self).__init__(
            pipelines,
            size,
            fill_last_batch,
            last_batch_padded,
            auto_reset)
        self._squeeze_labels = squeeze_labels
        self._dynamic_shape = dynamic_shape
        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()
        # Use double-buffering of data batches
        self._data_batches = [[None] for i in range(self._num_gpus)]
        self._current_data_batch = 0
        self._output_names_map = [x[0] for x in output_map]
        self._output_categories_map = [x[1] for x in output_map]
        self._output_categories = {DALIGenericIterator.DATA_TAG, DALIGenericIterator.LABEL_TAG}
        assert set(self._output_categories_map) <= self._output_categories, \
            "Only DATA_TAG and LABEL_TAG are allowed"
        assert len(set(self._output_names_map)) == len(self._output_names_map), \
            "output_names in output_map should be distinct"
        self.output_map = output_map

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.schedule_run()
        self._first_batch = None
        self._first_batch = self.next()
        # Set data descriptors for MXNet
        self.provide_data = []
        self.provide_label = []

        category_names = {key : [] for key in self._output_categories}
        for name, category in output_map:
            category_names[category].append(name)
        for i, data in enumerate(self._first_batch[0].data):
            data_shape  = (data.shape[0] * self._num_gpus,) + data.shape[1:]
            self.provide_data.append(mx.io.DataDesc(category_names[DALIGenericIterator.DATA_TAG][i], \
                data_shape, data.dtype, layout=data_layout))
        for i, label in enumerate(self._first_batch[0].label):
            label_shape = (label.shape[0] * self._num_gpus,) + label.shape[1:]
            self.provide_label.append(mx.io.DataDesc(category_names[DALIGenericIterator.LABEL_TAG][i], \
                label_shape, label.dtype))


    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        self._check_iteration_stop()
        # Gather outputs
        outputs = []
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            # MXNet wants batches with clear distinction between
            # data and label entries, so segregate outputs into
            # 2 categories
            category_outputs = {key : [] for key in self._output_categories}
            for j, out in enumerate(outputs[i]):
                category_outputs[self._output_categories_map[j]].append(out)
            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_info = dict()
            # For data proceed normally
            category_tensors[DALIGenericIterator.DATA_TAG] = \
                [x.as_tensor() for x in category_outputs[DALIGenericIterator.DATA_TAG]]
            category_info[DALIGenericIterator.DATA_TAG] = \
                [(x.shape(), np.dtype(x.dtype())) for x in category_tensors[DALIGenericIterator.DATA_TAG]]
            # For labels we squeeze the tensors
            category_tensors[DALIGenericIterator.LABEL_TAG] = \
                [x.as_tensor() for x in category_outputs[DALIGenericIterator.LABEL_TAG]]
            if self._squeeze_labels:
                for label in category_tensors[DALIGenericIterator.LABEL_TAG]:
                    label.squeeze()
            category_info[DALIGenericIterator.LABEL_TAG] = \
                [(x.shape(), np.dtype(x.dtype())) for x in category_tensors[DALIGenericIterator.LABEL_TAG]]

            # If we did not yet allocate memory for that batch, do it now
            if self._data_batches[i][self._current_data_batch] is None:
                mx_gpu_device = mx.gpu(self._pipes[i].device_id)
                mx_cpu_device = mx.cpu(0)
                from nvidia.dali.backend import TensorGPU
                category_device = {key : [] for key in self._output_categories}
                for category in self._output_categories:
                    for t in category_tensors[category]:
                        if type(t) is TensorGPU:
                            category_device[category].append(mx_gpu_device)
                        else:
                            category_device[category].append(mx_cpu_device)
                d = []
                l = []
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.DATA_TAG]):
                    d.append(mx.nd.zeros(shape, category_device[DALIGenericIterator.DATA_TAG][j], dtype = dtype))
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.LABEL_TAG]):
                    l.append(mx.nd.zeros(shape, category_device[DALIGenericIterator.LABEL_TAG][j], dtype = dtype))

                self._data_batches[i][self._current_data_batch] = mx.io.DataBatch(data=d, label=l)

            d = self._data_batches[i][self._current_data_batch].data
            l = self._data_batches[i][self._current_data_batch].label
            # Copy data from DALI Tensors to MXNet NDArrays
            if self._dynamic_shape:
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.DATA_TAG]):
                    if list(d[j].shape) != shape:
                        d[j] = mx.nd.zeros(shape, d[j].context, dtype = dtype)
                for j, (shape, dtype) in enumerate(category_info[DALIGenericIterator.LABEL_TAG]):
                    if list(l[j].shape) != shape:
                        l[j] = mx.nd.zeros(shape, l[j].context, dtype = dtype)

            for j, d_arr in enumerate(d):
                feed_ndarray(category_tensors[DALIGenericIterator.DATA_TAG][j], d_arr)
            for j, l_arr in enumerate(l):
                feed_ndarray(category_tensors[DALIGenericIterator.LABEL_TAG][j], l_arr)

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.release_outputs()
                p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 1
        self._counter += self._num_gpus * self.batch_size

        # padding the last batch
        if (not self._fill_last_batch) and (self._counter > self._size) and self._size > 0:
                # this is the last batch and we need to pad
                overflow = self._counter - self._size
                overflow_per_device = overflow // self._num_gpus
                difference = self._num_gpus - (overflow % self._num_gpus)
                for i in range(self._num_gpus):
                    if i < difference:
                        self._data_batches[i][copy_db_index].pad = overflow_per_device
                    else:
                        self._data_batches[i][copy_db_index].pad = overflow_per_device + 1
        else:
            for db in self._data_batches:
                db[copy_db_index].pad = 0

        return [db[copy_db_index] for db in self._data_batches]

    DATA_TAG = "data"
    LABEL_TAG = "label"

class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for MXNet. It returns 2 outputs
    (data and label) in the form of MXNet's DataBatch of NDArrays.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, size, data_name, label_name, data_layout)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines,
                           [(data_name, DALIClassificationIterator.DATA_TAG),
                            (label_name, DALIClassificationIterator.LABEL_TAG)],
                           size,
                           data_layout)

    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

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
    data_name : str, optional, default = 'data'
                Data name for provided symbols.
    label_name : str, optional, default = 'softmax_label'
                 Label name for provided symbols.
    data_layout : str, optional, default = 'NCHW'
                  Either 'NHWC' or 'NCHW' - layout of the pipeline outputs.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    squeeze_labels: bool, optional, default = True
                 Whether the iterator should squeeze the labels before
                 copying them to the ndarray.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the mxnet.ndarray will be resized accordingly
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
                 data_name='data',
                 label_name='softmax_label',
                 data_layout='NCHW',
                 fill_last_batch=True,
                 auto_reset=False,
                 squeeze_labels=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        super(DALIClassificationIterator, self).__init__(pipelines,
                                                         [(data_name, DALIClassificationIterator.DATA_TAG),
                                                          (label_name, DALIClassificationIterator.LABEL_TAG)],
                                                         size,
                                                         data_layout     = data_layout,
                                                         fill_last_batch = fill_last_batch,
                                                         auto_reset = auto_reset,
                                                         squeeze_labels=squeeze_labels,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded = last_batch_padded)

###############################################
###############################################
################## Gluon API ##################
###############################################
###############################################


class SmartArray(object):
    def __init__(self, array):
        self._data = array.reshape(-1)
        self._view = array

    def resize(self, shape):
        new_size = np.prod(shape)

        if new_size > self._data.size:
            self._data = mx.nd.zeros(new_size, self._data.context, dtype=self._data.dtype)
            self._view = self._data
        elif new_size < self._data.size:
            self._view = self._data[:new_size]
        else:
            self._view = self._data
        self._view = self._view.reshape(shape)
        return self._view

    @property
    def view(self):
        return self._view


class DALIGluonIterator(_DALIIteratorBase):
    """
    General DALI iterator for MXNet with Gluon API. It can return any number of
    outputs from the DALI pipeline in the form of per GPU tuples. These tuples consisting of
    NDArrays (for outputs marked as DALIGluonIterator.DENSE_TAG) and list of NDArrays (for
    output marked as DALIGluonIterator.SPARSE_TAG).



    Please keep in mind that NDArrays returned by the iterator are
    still owned by DALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another NDArray.

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
    output_types : list of str, optional, default = None
                 List of tags indicating whether the pipeline(s) output batch is
                 uniform (all the samples have the same size) or not. Batch output marked
                 as the former will be returned as a single NDArray, the latter
                 will be returned as a list of NDArray.
                 Must be either DALIGluonIterator.DENSE_TAG
                 or DALIGluonIterator.SPARSE_TAG.
                 Length of output_types must match the number of output of the pipeline(s).
                 If not set, all outputs are considered to be marked with DALIGluonIterator.DENSE_TAG.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
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
                 size,
                 output_types=None,
                 auto_reset=False,
                 fill_last_batch=True,
                 last_batch_padded=False):
        super(DALIGluonIterator, self).__init__(
            pipelines,
            size,
            fill_last_batch,
            last_batch_padded,
            auto_reset)

        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()
        self._data_batches = [None for i in range(self._num_gpus)]
        self._output_tags = {DALIGluonIterator.DENSE_TAG, DALIGluonIterator.SPARSE_TAG}
        assert output_types is None or set(output_types) <= self._output_tags, \
            "Only DENSE_TAG and SPARSE_TAG are allowed"

        self._outputs_types = output_types

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.schedule_run()


    def __next__(self):
        self._check_iteration_stop()
        # Gather outputs
        dali_outputs = []
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                dali_outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            output_elements = []
            shapes = []
            for j, out in enumerate(dali_outputs[i]):
                if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                    output_elements.append(out.as_tensor())
                    shapes.append(output_elements[-1].shape())
                else:
                    output_elements.append([out[sample_idx] for sample_idx in range(self.batch_size)])
                    s = [t.shape() for t in output_elements[-1]]
                    shapes.append(s)

            if self._data_batches[i] is None:
                self._data_batches[i] = self._create_data_batch(output_elements, shapes, self._pipes[i].device_id)

            batch = self._data_batches[i]
            # Copy data from DALI Tensors to MXNet NDArrays
            for j, output_el in enumerate(output_elements):
                if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                    ndarray = batch[j].resize(shapes[j])
                    feed_ndarray(output_el, ndarray)
                else:
                    for sample_idx in range(self.batch_size):
                        ndarray = batch[j][sample_idx].resize(shapes[j][sample_idx])
                        feed_ndarray(output_el[sample_idx], ndarray)

        batches = [[([sample.view for sample in output_el] if isinstance(output_el,list) else output_el.view)
                    for output_el in batch]
                   for batch in self._data_batches]

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.release_outputs()
                p.schedule_run()

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
            output = batches[0:numGPUs_tograb]
            output[-1] = output[-1].copy()
            for element_idx in range(len(output[-1])):
                output[-1][element_idx] = output[-1][element_idx][0:data_fromlastGPU]
            return output

        return batches

    def _create_data_batch(self, output_elements, shapes, device_id):
        mx_gpu_device = mx.gpu(device_id)
        mx_cpu_device = mx.cpu(0)
        from nvidia.dali.backend import TensorGPU
        new_batch = []
        for j, output_el in enumerate(output_elements):
            first_t = output_el if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG else output_el[0]
            dtype = np.dtype(first_t.dtype())
            device = mx_gpu_device if type(first_t) is TensorGPU else mx_cpu_device
            if self._outputs_types is None or self._outputs_types[j] == DALIGluonIterator.DENSE_TAG:
                new_batch.append(SmartArray(mx.nd.zeros(shapes[j], device, dtype=dtype)))
            else:
                l = []
                for sample_idx in range(self.batch_size):
                    l.append(SmartArray(mx.nd.zeros(shapes[j][sample_idx], device, dtype=dtype)))
                new_batch.append(l)
        return new_batch

    DENSE_TAG = "dense"
    SPARSE_TAG = "sparse"
