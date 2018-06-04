from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ndll.pipeline import Pipeline
import mxnet as mx
import ctypes
import logging

# MXNet currently does not expose WaitToWrite C API call
# in Python API
def _wait_to_write(arr):
    if not isinstance(arr, mx.nd.NDArray):
        raise RuntimeError("Can only wait for NDArray")
    mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)

def feed_ndarray(ndll_tensor, arr):
    # Wait until arr is no longer used by the engine
    _wait_to_write(arr)
    assert ndll_tensor.shape() == list(arr.shape), \
            ("Shapes do not match: NDLL tensor has shape {0}"
            ", but NDArray has shape {1}".format(ndll_tensor.shape(), list(arr.shape)))
    # Get CTypes void pointer to the underlying memory held by arr
    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr.handle, ctypes.byref(ptr))
    # Copy data from NDLL tensor to ptr
    ndll_tensor.copy_to_external(ptr)

class NDLLGenericIterator:
    def __init__(self,
                 pipelines,
                 output_map,
                 size = -1,
                 data_name='data',
                 label_name='softmax_label',
                 data_layout='NCHW'):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = size
        self._pipes = pipelines
        # Build all pipelines
        for p in self._pipes:
            p.build()
        # Use double-buffering of data batches
        self._data_batches = [[None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = output_map

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = self.next()
        # Set data descriptors for MXNet
        self.provide_data = []
        self.provide_label = []
        for data in self._first_batch[0].data:
            data_shape  = (data.shape[0] * self._num_gpus,) + data.shape[1:]
            self.provide_data.append(mx.io.DataDesc(data_name, data_shape, data.dtype, layout=data_layout))
        for label in self._first_batch[0].label:
            label_shape = (label.shape[0] * self._num_gpus,) + label.shape[1:]
            self.provide_label.append(mx.io.DataDesc(label_name, label_shape, label.dtype))


    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration
        # Gather outputs
        outputs = []
        for p in self._pipes:
            outputs.append(p.run())
        for i in range(self._num_gpus):
            out_data = []
            out_label = []
            # MXNet wants batches with clear distinction between
            # data and label entries, so segregate outputs into
            # 2 categories
            for i, out in enumerate(outputs[i]):
                if self.output_map[i] == "data":
                    out_data.append(out)
                elif self.output_map[i] == "label":
                    out_label.append(out)

            # Change NDLL TensorLists into Tensors
            data = list(map(lambda x: x.as_tensor(), out_data))
            data_shape = list(map(lambda x: x.shape(), data))
            label = list(map(lambda x: x.as_tensor(), out_label))
            # Change label shape from [batch_size, 1] to [batch_size]
            for l in label:
                l.squeeze()
            label_shape = list(map(lambda x: x.shape(), label))
            # If we did not yet allocate memory for that batch, do it now
            if self._data_batches[i][self._current_data_batch] is None:
                d = [mx.nd.zeros(shape, mx.gpu(self._pipes[i].device_id)) for shape in data_shape]
                l = [mx.nd.zeros(shape, mx.cpu(0)) for shape in label_shape]
                self._data_batches[i][self._current_data_batch] = mx.io.DataBatch(data=d, label=l)
            d = self._data_batches[i][self._current_data_batch].data
            l = self._data_batches[i][self._current_data_batch].label
            # Copy data from NDLL Tensors to MXNet NDArrays
            for i, d_arr in enumerate(d):
                feed_ndarray(data[i], d_arr)
            for i, l_arr in enumerate(l):
                feed_ndarray(label[i], l_arr)
        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        return [db[copy_db_index] for db in self._data_batches]

    next = __next__

    def __iter__(self):
        return self

    def reset(self):
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warn("NDLL iterator does not support resetting while epoch is not finished. Ignoring...")

class NDLLClassificationIterator(NDLLGenericIterator):
    def __init__(self,
                 pipelines,
                 size = -1,
                 data_name='data',
                 label_name='softmax_label',
                 data_layout='NCHW'):
        super(NDLLClassificationIterator, self).__init__(pipelines, ["data", "label"],
                                                         size, data_name, label_name,
                                                         data_layout)
