from nvidia.dali.backend_impl import *
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mxnet as mx
import ctypes

def test_copy_to_external_mxnet():
    arr = np.random.rand(1000, 1000, 3)
    tensor = TensorGPU(arr)
    assert_array_equal(arr, np.array(tensor))

    buf = mx.nd.zeros((1000,1000,3), mx.gpu(0))
    mx.base._LIB.MXNDArrayWaitToWrite(buf.handle)

    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(buf.handle, ctypes.byref(ptr))
    tensor.copy_to_external(ptr)

    assert_array_equal(np.array(tensor), buf.asnumpy())

test_copy_to_external_mxnet()
