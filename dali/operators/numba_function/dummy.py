import numpy as np
import time
from numba import cfunc, types, carray, objmode

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

c_sig = types.void(types.CPointer(types.uint8),
                   types.CPointer(types.uint8),
                   types.int64)

@cfunc(c_sig, nopython=True)
def hello_cfunc(out_ptr, in_ptr, size):
    in_arr = carray(in_ptr, size)
    out_arr = carray(out_ptr, size)

    out_arr[:] = 255

def get_data(shapes):
    return [np.empty(shape, dtype = np.uint8) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, fn_ptr=None):
    data = fn.external_source(lambda: get_data(shapes), batch=True, device = "cpu")
    return fn.numba_func(data, fn_ptr=fn_ptr)

def _testimpl_numba_func(shapes, fn_ptr):
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, fn_ptr=fn_ptr)
    pipe.build()
    outs = pipe.run()
    for i in range(batch_size):
        out_arr = np.array(outs[0][i])
        print(out_arr)

_testimpl_numba_func([[10, 10, 10]], hello_cfunc.address)