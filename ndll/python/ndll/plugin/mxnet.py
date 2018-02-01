from __future__ import absolute_import
from ndll.pipeline import Pipeline
import mxnet as mx
import ctypes

class NDLLIterator:
    def __init__(self,
                 batch_size,
                 pipeline,
                 num_gpus,
                 num_threads,
                 size,
                 data_name='data',
                 label_name='softmax_label'):
        self._s = pipeline.serialize()
        self._batch_size = batch_size
        self._num_gpus = num_gpus
        self._num_threads = num_threads
        self._size = size
        self._pipes = [Pipeline(batch_size, num_threads, i, True, True) for i in range(num_gpus)]
        for p in self._pipes:
            p.deserialize_and_build(self._s)
        self._data_batches = [[None, None] for i in range(num_gpus)]
        self._counter = 0
        self._current_data_batch = 0

    def __next__(self):
        if self._counter > self._size:
            self._counter = self._counter % self._size
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
            d_ptr = ctypes.c_void_p()
            l_ptr = ctypes.c_void_p()
            mx.base._LIB.MXNDArrayGetData(d[0].handle, ctypes.byref(d_ptr))
            mx.base._LIB.MXNDArrayGetData(l[0].handle, ctypes.byref(l_ptr))
            data.copy_to_external(d_ptr)
            label.copy_to_external(l_ptr)
        copy_db_index = self._current_data_batch
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self._batch_size
        return [db[copy_db_index] for db in self._data_batches]


    next = __next__

    def __iter__(self):
        return self
