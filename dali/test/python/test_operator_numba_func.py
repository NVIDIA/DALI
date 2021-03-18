import numpy as np
import os
from numba import cfunc, types, carray

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

fun1_sig = types.void(types.CPointer(types.uint8),
                    types.CPointer(types.uint8),
                    types.int64)
@cfunc(fun1_sig, nopython=True)
def set_all_values_to_255(out_ptr, in_ptr, size):
    out_arr = carray(out_ptr, size)
    out_arr[:] = 255

fun2_sig = types.void(types.CPointer(types.float32),
                    types.CPointer(types.float32),
                    types.int64)
@cfunc(fun2_sig, nopython=True)
def set_all_values_to_float(out_ptr, in_ptr, size):
    out_arr = carray(out_ptr, size)
    out_arr[:] = 0.5

def get_data(shapes, dtype):
    return [np.empty(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, fn_ptr=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return fn.numba_func(data, fn_ptr=fn_ptr)

def _testimpl_numba_func(shapes, dtype, fn_ptr, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype, fn_ptr=fn_ptr)
    pipe.build()
    outs = pipe.run()
    for _ in range(3):
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert np.array_equal(out_arr, expected_out)

def test_numba_func():
    # shape, dtype, func address, expected_out
    args = [
        ([(10, 10, 10)], np.uint8, set_all_values_to_255.address, np.full((10, 10, 10), 255, dtype=np.uint8)),
        ([(10, 10, 10)], np.float32, set_all_values_to_float.address, np.full((10, 10, 10), 0.5, dtype=np.float32)),
    ]

    for shape, dtype, fn_ptr, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, fn_ptr, expected_out

@pipeline_def
def numba_func_image_pipe(fn_ptr=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    images_out = dali.fn.numba_func(images_in, fn_ptr=fn_ptr)
    return images_in, images_out

def _testimpl_numba_func_image(fn_ptr, transform):
    pipe = numba_func_image_pipe(batch_size=8, num_threads=1, device_id=0, fn_ptr=fn_ptr)
    pipe.build()
    for _ in range(3):
        images_in, images_out = pipe.run()
        for i in range(len(images_in)):
            image_in_transformed = transform(images_in.at(i))
            assert np.array_equal(image_in_transformed, images_out.at(i))

@cfunc(fun1_sig, nopython=True)
def reverse_col(out_ptr, in_ptr, size):
    in_arr = carray(in_ptr, size)
    out_arr = carray(out_ptr, size)
    for i in range(size):
        out_arr[i] = 255 - in_arr[i]

def test_numba_func_image():
    args = [
        (reverse_col.address, lambda x: 255 - x),
    ]
    for fn_ptr, transform in args:
        yield _testimpl_numba_func_image, fn_ptr, transform