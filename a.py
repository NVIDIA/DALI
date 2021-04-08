from nvidia.dali import types as dali_types
import numba as nb
from numba import njit, carray, cfunc
from numba.core import types
import numpy as np
import ctypes

arr = np.arange(5).astype(np.double)

@cfunc(types.void(types.CPointer(types.void)), nopython=True)
def abc(x):
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((20, 10), dtype=np.uint8)
    z = []
    z.append(a)
    z.append(b)
    
abc(42)