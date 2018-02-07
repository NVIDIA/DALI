from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ndll.pipeline import Pipeline
import mxnet as mx
import ctypes

# MXNet currently does not expose WaitToWrite C API call
# in Python API
def _wait_to_write(arr):
    if not isinstance(arr, mx.nd.NDArray):
        raise RuntimeError("Can only wait for NDArray")
    mx.base._LIB.MXNDArrayWaitToWrite(arr.handle)

def feed_ndarray(ndll_tensor, arr):
    _wait_to_write(arr)
    assert ndll_tensor.shape() == list(arr.shape), \
            ("Shapes do not match: NDLL tensor has shape {0}"
            ", but NDArray has shape {1}".format(ndll_tensor.shape(), list(arr.shape)))
    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr.handle, ctypes.byref(ptr))
    ndll_tensor.copy_to_external(ptr)


class NDLLIterator:
    def __init__(self,
                 batch_size,
                 pipeline,
                 num_gpus = 1,
                 num_threads = 2,
                 size = -1,
                 data_name='data',
                 label_name='softmax_label'):
        self._s = pipeline.serialize()
        self.batch_size = batch_size
        self._num_gpus = num_gpus
        assert num_gpus > 0, "Number of GPUs used has to be at least 1"
        self._num_threads = num_threads
        self._size = size
        self._pipes = [Pipeline(batch_size, num_threads, i, True, True) for i in range(num_gpus)]
        for p in self._pipes:
            p.deserialize_and_build(self._s)
        self._data_batches = [[None, None] for i in range(num_gpus)]
        self._counter = 0
        self._current_data_batch = 0

        self._first_batch = None
        self._first_batch = self.next()
        data = self._first_batch[0].data[0]
        label = self._first_batch[0].label[0]

        data_shape  = (data.shape[0] * num_gpus,) + data.shape[1:]
        label_shape = (label.shape[0] * num_gpus,) + label.shape[1:]

        self.provide_data = [mx.io.DataDesc(data_name, data_shape, data.dtype)]
        self.provide_label = [mx.io.DataDesc(label_name, label_shape, label.dtype)]

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration
        outputs = []
        for p in self._pipes:
            outputs.append(p.run())
        for i in range(self._num_gpus):
            data, label = outputs[i]
            data = data.as_tensor()
            data_shape = data.shape()
            label = label.as_tensor()
            label.squeeze()
            label_shape = label.shape()
            if self._data_batches[i][self._current_data_batch] is None:
                d = mx.nd.zeros(data_shape, mx.gpu(0))
                l = mx.nd.zeros(label_shape, mx.cpu(0))
                self._data_batches[i][self._current_data_batch] = mx.io.DataBatch(data=[d], label=[l])
            d = self._data_batches[i][self._current_data_batch].data
            l = self._data_batches[i][self._current_data_batch].label
            feed_ndarray(data, d[0])
            feed_ndarray(label, l[0])
        copy_db_index = self._current_data_batch
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
            raise RuntimeWarning("NDLL iterator does not support resetting while epoch is not finish. Ignoring...")
