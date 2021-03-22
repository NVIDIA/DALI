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
                    types.CPointer(types.int64),
                    types.CPointer(types.int64),
                    types.int64)
@cfunc(fun1_sig, nopython=True)
def set_all_values_to_255(out_ptr, in_ptr, out_shape_ptr, in_shape_ptr, ndim):
    out_shape = carray(out_shape_ptr, ndim)
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 255

fun2_sig = types.void(types.CPointer(types.float32),
                    types.CPointer(types.float32),
                    types.CPointer(types.int64),
                    types.CPointer(types.int64),
                    types.int64)
@cfunc(fun2_sig, nopython=True)
def set_all_values_to_float(out_ptr, in_ptr, out_shape_ptr, in_shape_ptr, ndim):
    out_shape = carray(out_shape_ptr, ndim)
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 0.5

setup1_sig = types.void(types.CPointer(types.int64),
                types.CPointer(types.int64),
                types.int64,
                types.int64,
                types.CPointer(types.int64),
                types.CPointer(types.int64),
                types.int64,
                types.int64)
@cfunc(setup1_sig, nopython=True)
def setup_fn1(out_shape_ptr, in_shape_ptr, num_samples, ndim, out_type_ptr, in_type_ptr, num_outputs, num_inputs):
    in_arr = carray(in_shape_ptr, num_samples * ndim)
    out_arr = carray(out_shape_ptr, num_samples * ndim)
    out_arr[0] = in_arr[1]
    out_arr[1] = in_arr[2]
    out_arr[2] = in_arr[0]
    out_arr[3] = in_arr[5]
    out_arr[4] = in_arr[3]
    out_arr[5] = in_arr[4]
    in_type = carray(in_type_ptr, 1)
    out_type = carray(out_type_ptr, 1)
    out_type[0] = 6

fun3_sig = types.void(types.CPointer(types.int32),
            types.CPointer(types.int32),
            types.CPointer(types.int64),
            types.CPointer(types.int64),
            types.int64)
@cfunc(fun3_sig, nopython=True)
def fun3(out_ptr, in_ptr, out_shape_ptr, in_shape_ptr, ndim):
    out_shape = carray(out_shape_ptr, ndim)
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 42

def get_data(shapes, dtype):
    return [np.empty(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, fn_ptr=None, setup_fn=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return fn.experimental.numba_func(data, fn_ptr=fn_ptr, setup_fn=setup_fn)

def _testimpl_numba_func(shapes, dtype, fn_ptr, setup_fn, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype, fn_ptr=fn_ptr, setup_fn=setup_fn)
    pipe.build()
    outs = pipe.run()
    for _ in range(3):
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert np.array_equal(out_arr, expected_out[i])

def test_numba_func():
    # shape, dtype, func address, expected_out
    args = [
        ([(10, 10, 10)], np.uint8, set_all_values_to_255.address, None, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        ([(10, 10, 10)], np.float32, set_all_values_to_float.address, None, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        ([(10, 20, 30), (20, 10, 30)], np.int64, fun3.address, setup_fn1.address, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((30, 20, 10), 42, dtype=np.int32)]),
    ]

    for shape, dtype, fn_ptr, setup_fn, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, fn_ptr, setup_fn, expected_out

@pipeline_def
def numba_func_image_pipe(fn_ptr=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    images_out = dali.fn.experimental.numba_func(images_in, fn_ptr=fn_ptr)
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
def reverse_col(out_ptr, in_ptr, out_shape_ptr, in_shape_ptr, ndim):
    out_shape = carray(out_shape_ptr, ndim)
    in_shape = carray(in_shape_ptr, ndim)
    in_arr = carray(in_ptr, (in_shape[0], in_shape[1], in_shape[2]))
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 255 - in_arr[:]

def test_numba_func_image():
    args = [
        (reverse_col.address, lambda x: 255 - x),
    ]
    for fn_ptr, transform in args:
        yield _testimpl_numba_func_image, fn_ptr, transform
