from numba import njit, carray, cfunc
from numba import types as numba_types
import numpy as np
import numba as nb
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

@cfunc(_run_fn_sig(), nopython=True)
def help(a, b, c, d):
    x = _get_shape_view(a, b, c, d)
    x[0][0][1] = 30
    # y = address_as_void_pointer(a)

print(y, cc.ctypes.data)
help(y.ctypes.data, ndim.ctypes.data, 1, 1)
print(cc)