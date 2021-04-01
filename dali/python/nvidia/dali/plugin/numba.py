from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from numba import types
import nvidia.dali.types as dali_types
import math
import numpy as np

to_dali_type = {
    types.bool__ : dali_types.BOOL,
    types.int__ : dali_types.INT64,
    types.uint__ : dali_types.UINT64,
    types.int8 : dali_types.INT8,
    types.int8 : dali_types.INT8,
    types.int16 : dali_types.INT16,
    types.int32 : dali_types.INT32,
    types.int64 : dali_types.INT64,
    types.uint16 : dali_types.UINT16,
    types.uint32 : dali_types.UINT32,
    types.uint64 : dali_types.UINT64,
    types.float32 : dali_types.FLOAT,
    types.float64 : dali_types.FLOAT64,
}

class NumbaFunc(ops.PythonFunctionBase):
    ops.register_cpu_op('NumbaFunc')

    def _setup_fn_sig():
        return types.void(types.CPointer(types.CPointer(types.int64)),
                        types.CPointer(types.int64),
                        types.int32,
                        types.CPointer(types.CPointer(types.int64)),
                        types.CPointer(types.int64),
                        types.int32, types.int32)

    @staticmethod
    @njit
    def _get_shape_view(shapes_ptr, ndims_ptr, num_dims, num_samples):
        ndims = carray(ndims_ptr, num_dims)
        samples = carray(shapes_ptr, (num_samples, num_dims))
        return np.array([np.array([carray(shape_ptr, size)]) for sample in samples for shape_ptr, size in zip(sample, ndims)])

    @staticmethod
    @njit
    def _get_view(arr_ptr, arr_types, shapes, num_dims, num_samples):
        # ret[0] - sample
        # ret[0][0] - output 0 of sample 0
        @njit
        def get_carray(ptr, shape, arr_type):
            if len(shape) == 1:
                return carray(arr_ptr, shape[0], dtype=arr_type)
            elif len(shape) == 2:
                return carray(arr_ptr, (shape[0], shape[1]), dtype=arr_type)
            elif len(shape) == 3:
                return carray(arr_ptr, (shape[0], shape[1], shape[2]), dtype=arr_type)
            elif len(shape) == 4:
                return carray(arr_ptr, (shape[0], shape[1], shape[2], shape[3]), dtype=arr_type)
            elif len(shape) == 5:
                return carray(arr_ptr, (shape[0], shape[1], shape[2], shape[3], shape[4]), dtype=arr_type)
            elif len(shape) == 6:
                return carray(arr_ptr, (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]), dtype=arr_type)
            assert False # todo

        samples = carray(arr_ptr, (num_samples, num_dims))
        return np.array([np.array([get_carray(arr_ptr, shape, arr_type)] for sample, sample_shape in zip(samples, shapes) for arr_ptr, shape, arr_type in zip(sample, sample_shape, arr_types))])
                
    def _run_fn_sig(out_types, in_types):
        sig_types = []
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.int32)

        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.CPointer(types.int64))
        sig_types.append(types.int32)

        sig_types.append(types.int32)

        return types.void(*sig_types)
        

    def __init__(self, run_fn, out_types, in_types, setup_fn=None, num_inputs=1, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        out_types_dali = [to_dali_type[numba_type] for numba_type in out_types]
        in_types_dali = [to_dali_type[numba_type] for numba_type in in_types]

        setup_fn = njit(setup_fn)
        @cfunc(_setup_fn_sig(), nopython=True)
        def setup_cfunc(out_shapes_ptr, out_ndims_ptr, num_outs, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
            out_shapes_np = _get_shape_view(out_shape_ptr, out_ndims_ptr, num_outs, num_samples)
            in_shapes_np = _get_shape_view(in_shape_ptr, in_ndims_ptr, num_outs, num_samples)
            setup_fn(out_shapes_np, in_shapes_np)

        run_fn = njit(run_fn)
        @cfunc(_run_fn_sig(), nopython=True)
        def run_cfunc(out_ptr, out_types_ptr, out_shapes_ptr, out_ndims_ptr, num_outs, in_ptr, in_types_ptr, in_shapes_ptr, in_ndim_ptr, num_ins, num_samples):
            out_shapes_np = _get_shape_view(out_shape_ptr, out_ndims_ptr, num_outs, num_samples)
            out_types = carray(out_types_ptr, num_outs)
            outs = _get_view(out_ptr, out_types, out_shapes_np, num_outs, num_samples)
            
            in_shapes_np = _get_shape_view(in_shape_ptr, in_ndims_ptr, num_ins, num_samples)
            in_types = carray(in_types_ptr, num_ins)
            ins = _get_view(in_ptr, in_types, in_shapes_np, num_ins, num_samples)

            run_fn(outs, ins)

        super(NumbaFunc, self).__init__(impl_name="NumbaFuncImpl",
                                                setup_fn=setup_cfunc.address, run_fn=run_cfunc.address,
                                                out_types_dali=out_types_dali, in_types_dali_dali=in_types,
                                                out_types_numba=out_types, in_types_numba=in_types,
                                                num_inputs=num_inputs, num_outputs=num_outputs,
                                                device=device, batch_processing=batch_processing, **kwargs)

ops._wrap_op(NumbaFunc, [], __name__)