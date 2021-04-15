from numba import njit, carray, cfunc
from numba import types as numba_types
import numpy as np
import numba as nb
from nvidia.dali import types as dali_types
import ctypes

@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

@njit
def _get_shape_view(shapes_ptr, ndims_ptr, num_dims, num_samples):
    ndims = carray(address_as_void_pointer(ndims_ptr), num_dims, dtype=np.int32)
    samples = carray(address_as_void_pointer(shapes_ptr), (num_dims, num_samples), dtype=np.int64)
    l = []
    for sample, size in zip(samples, ndims):
        d = []
        for shape_ptr in sample:
            d.append(carray(address_as_void_pointer(shape_ptr), size, dtype=np.int64))
        l.append(d)
    return l

def _run_fn_sig():
    sig_types = []
    sig_types.append(numba_types.int64)
    sig_types.append(numba_types.int64)
    sig_types.append(numba_types.int32)
    sig_types.append(numba_types.int32)
    return numba_types.void(*sig_types)

x = np.zeros((2, 2, 2), dtype=np.uint8)
cc = np.array(x.shape, dtype=np.int64)
y = np.array([cc.ctypes.data])
ndim = np.array([3], dtype=np.int64)

def _get_carray_eval_lambda(n, m):
        eval_string = "lambda "
        for i in range(n):
            eval_string += "out{}".format(i)
            eval_string += ", "
        for i in range(m):
            eval_string += "in{}".format(i)
            eval_string += ", " if i + 1 != m else ": "

        eval_string += "run_fn(["
        for i in range(n):
            eval_string += "out{}".format(i)
            eval_string += ", " if i + 1 != n else  "], "
        eval_string += "["
        for i in range(m):
            eval_string += "in{}".format(i)
            eval_string += ", " if i + 1 != n else  "]"
        eval_string += ")"
        print(eval_string)
        return njit(eval(eval_string))

def _get_carray_eval_lambda(dtype, ndim):
    eval_string = "lambda ptr, shape: carray(ptr, ("
    for i in range(ndim):
        eval_string += "shape[{}]".format(i)
        eval_string += ", " if i + 1 != ndim else "), "
    eval_string += "dtype=np.{})".format(dtype.name.lower())
    return njit(eval(eval_string))

def _get_pass_lambda():
    eval_string = "lambda x, y: x"
    return njit(eval(eval_string))

def _get_carrays_eval_lambda(types, ndim):
    return tuple([_get_carray_eval_lambda(dtype, ndim) for dtype, ndim in zip(types, ndim)] + [njit(eval(("lambda x, y: 0"))) for i in range(6 - len(types))])

out0_lambda, out1_lambda, out2_lambda, out3_lambda, out4_lambda, out5_lambda = _get_carrays_eval_lambda([dali_types.INT8], [3])
print(out1_lambda)