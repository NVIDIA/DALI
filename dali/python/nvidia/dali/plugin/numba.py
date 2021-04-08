from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops
from nvidia.dali import types as dali_types
from numba import types as numba_types
from numba import njit, cfunc, carray
import numpy as np
import numba as nb
import ctypes

to_numba_type = {
    int(dali_types.INT8) : numba_types.int8,
    int(dali_types.INT8) : numba_types.int8,
    int(dali_types.INT16) : numba_types.int16,
    int(dali_types.INT32) : numba_types.int32,
    int(dali_types.INT64) : numba_types.int64,
    int(dali_types.UINT16) : numba_types.uint16,
    int(dali_types.UINT32) : numba_types.uint32,
    int(dali_types.UINT64) : numba_types.uint64,
    int(dali_types.FLOAT) : numba_types.float32,
    int(dali_types.FLOAT64) : numba_types.float64,
}

@njit
def _get_shape_view(shapes_ptr, ndims_ptr, num_dims, num_samples):
    ndims = carray(ndims_ptr, num_dims)
    samples = carray(shapes_ptr, (num_samples, num_dims))
    l = []
    for sample in samples:
        d = []
        for shape_ptr, size in zip(sample, ndims):
            d.append(carray(shape_ptr, size))
        l.append(d)
    return l

@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    from numba.core import types, cgutils
    sig = types.voidptr(src)
    print(type(sig))

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

@njit
def _get_view(arr_ptr, arr_types, shapes, num_dims, num_samples):
    # ret[0] - sample
    # ret[0][0] - output 0 of sample 0
    samples = carray(arr_ptr, (num_samples, num_dims))
    l = []
    for sample, sample_shape in zip(samples, shapes):
        d = []
        for ptr, shape, arr_type in zip(sample, sample_shape, arr_types):
            void_ptr = address_as_void_pointer(ptr)
            if len(shape) == 1:
                d.append(carray(void_ptr, shape[0], dtype=np.uint8))
            elif len(shape) == 2:
                d.append(carray(void_ptr, (shape[0], shape[1]), dtype=np.uint8))
            elif len(shape) == 3:
                d.append(carray(void_ptr, (shape[0], shape[1], shape[2]), dtype=np.uint8))
            elif len(shape) == 4:
                d.append(carray(void_ptr, (shape[0], shape[1], shape[2], shape[3]), dtype=np.uint8))
            elif len(shape) == 5:
                d.append(carray(void_ptr, (shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=np.uint8))
            elif len(shape) == 6:
                d.append(carray(void_ptr, (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]), dtype=np.uint8))
        l.append(d)
    return l

class NumbaFunc(ops.PythonFunctionBase):
    ops.register_cpu_op('NumbaFunc')

    def _setup_fn_sig(self):
        return numba_types.void(numba_types.CPointer(numba_types.CPointer(numba_types.int64)),
                        numba_types.CPointer(numba_types.int64),
                        numba_types.int32,
                        numba_types.CPointer(numba_types.CPointer(numba_types.int64)),
                        numba_types.CPointer(numba_types.int64),
                        numba_types.int32, numba_types.int32)
                
    def _run_fn_sig(self):
        sig_types = []
        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.void)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.int64)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.int32)

        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.void)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.int64)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.int32)

        sig_types.append(numba_types.int32)

        return numba_types.void(*sig_types)

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if pipeline is None:
            Pipeline._raise_no_current_pipeline("NumbaFunc")
        return super(NumbaFunc, self).__call__(*inputs, **kwargs)

    def __init__(self, run_fn, out_types, in_types, outs_ndim=None, setup_fn=None, num_inputs=1, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        # TODO, add batch support
        setup_fn_address = None
        if setup_fn != None:
            setup_fn = njit(setup_fn)
            @cfunc(self._setup_fn_sig(), nopython=True)
            def setup_cfunc(out_shapes_ptr, out_ndims_ptr, num_outs, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
                out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, num_samples)
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_outs, num_samples)
                setup_fn(out_shapes_np, in_shapes_np)
            setup_fn_address = setup_cfunc.address

        run_fn = njit(run_fn)
        @cfunc(self._run_fn_sig(), nopython=True)
        def run_cfunc(out_ptr, out_types_ptr, out_shapes_ptr, out_ndims_ptr, num_outs, in_ptr, in_types_ptr, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
            out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, num_samples)
            out_types = carray(out_types_ptr, num_outs)
            outs = _get_view(out_ptr, out_types, out_shapes_np, num_outs, num_samples)
            
            in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_ins, num_samples)
            in_types = carray(in_types_ptr, num_ins)
            ins = _get_view(in_ptr, in_types, in_shapes_np, num_ins, num_samples)

            run_fn(outs, ins)

        super(NumbaFunc, self).__init__(impl_name="NumbaFuncImpl",
                                                setup_fn=setup_fn_address, run_fn=run_cfunc.address,
                                                out_types=out_types, in_types=in_types,
                                                num_inputs=num_inputs, num_outputs=num_outputs, outs_ndim=outs_ndim,
                                                device=device, batch_processing=batch_processing, **kwargs)

ops._wrap_op(NumbaFunc, "fn", __name__)