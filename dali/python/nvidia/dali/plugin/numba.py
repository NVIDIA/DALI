from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops
from nvidia.dali import types as dali_types
from numba import types as numba_types
from numba import njit, cfunc, carray
import numpy as np
import numba as nb
import ctypes

@njit
def _get_shape_view(shapes_ptr, ndims_ptr, num_dims, num_samples):
    ndims = carray(ndims_ptr, num_dims)
    samples = carray(shapes_ptr, (num_dims, num_samples))
    l = []
    for sample, size in zip(samples, ndims):
        d = []
        for shape_ptr in sample:
            d.append(carray(shape_ptr, size))
        l.append(d)
    return l

@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen

class NumbaFunc(ops.NumbaFunctionBase):
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
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.int64)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.int32)

        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.CPointer(numba_types.CPointer(numba_types.int64)))
        sig_types.append(numba_types.CPointer(numba_types.int64))
        sig_types.append(numba_types.int32)

        sig_types.append(numba_types.int32)

        return numba_types.void(*sig_types)

    def _get_carray_eval_lambda(self, dtype, ndim):
        eval_string = "lambda ptr, shape: carray(ptr, ("
        for i in range(ndim):
            eval_string += "shape[{}]".format(i)
            eval_string += ", " if i + 1 != ndim else "), "
        eval_string += "dtype=np.{})".format(dtype.name.lower())
        return njit(eval(eval_string))

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if pipeline is None:
            Pipeline._raise_no_current_pipeline("NumbaFunc")
        return super(NumbaFunc, self).__call__(*inputs, **kwargs)

    def __init__(self, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn=None, num_inputs=1, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
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

        out0_get_carray = self._get_carray_eval_lambda(out_types[0], outs_ndim[0])
        in0_get_carray = self._get_carray_eval_lambda(in_types[0], ins_ndim[0])
        run_fn = njit(run_fn)
        @cfunc(self._run_fn_sig(), nopython=True)
        def run_cfunc(out_ptr, out_types_ptr, out_shapes_ptr, out_ndims_ptr, num_outs, in_ptr, in_types_ptr, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
            out0 = out1 = out2 = out3 = out4 = out5 = None
            out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, num_samples)
            out_types = carray(out_types_ptr, num_outs)
            out_arr = carray(out_ptr, (num_outs, num_samples))
            out0 = out0_get_carray(address_as_void_pointer(out_arr[0][0]), out_shapes_np[0][0])
            
            in0 = in1 = in2 = in3 = in4 = in5 = None
            in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_ins, num_samples)
            in_types = carray(in_types_ptr, num_ins)
            in_arr = carray(in_ptr, (num_ins, num_samples))
            in0 = in0_get_carray(address_as_void_pointer(in_arr[0][0]), in_shapes_np[0][0])

            invoke_run_fn(run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5, num_ins, num_outs)

        super(NumbaFunc, self).__init__(impl_name="NumbaFuncImpl",
                                                setup_fn=setup_fn_address, run_fn=run_cfunc.address,
                                                out_types=out_types, in_types=in_types, outs_ndim=outs_ndim,
                                                device=device, batch_processing=batch_processing, **kwargs)

@njit
def invoke_run_fn(run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5, num_ins, num_outs):
    if num_outs == 1 and num_ins == 1:
        run_fn(out0, in0)
    elif num_outs == 1 and num_ins == 2:
        run_fn(out0, in0, in1)
    elif num_outs == 1 and num_ins == 3:
        run_fn(out0, in0, in1, in2)
    elif num_outs == 1 and num_ins == 4:
        run_fn(out0, in0, in1, in2, in3)
    elif num_outs == 1 and num_ins == 5:
        run_fn(out0, in0, in1, in2, in3, in4)
    elif num_outs == 1 and num_ins == 6:
        run_fn(out0, in0, in1, in2, in3, in4, in5)
    elif num_outs == 2 and num_ins == 1:
        run_fn(out0, out1, in0)
    elif num_outs == 2 and num_ins == 2:
        run_fn(out0, out1, in0, in1)
    elif num_outs == 2 and num_ins == 3:
        run_fn(out0, out1, in0, in1, in2)
    elif num_outs == 2 and num_ins == 4:
        run_fn(out0, out1, in0, in1, in2, in3)
    elif num_outs == 2 and num_ins == 5:
        run_fn(out0, out1, in0, in1, in2, in3, in4)
    elif num_outs == 2 and num_ins == 6:
        run_fn(out0, out1, in0, in1, in2, in3, in4, in5)
    elif num_outs == 3 and num_ins == 1:
        run_fn(out0, out1, out2, in0)
    elif num_outs == 3 and num_ins == 2:
        run_fn(out0, out1, out2, in0, in1)
    elif num_outs == 3 and num_ins == 3:
        run_fn(out0, out1, out2, in0, in1, in2)
    elif num_outs == 3 and num_ins == 4:
        run_fn(out0, out1, out2, in0, in1, in2, in3)
    elif num_outs == 3 and num_ins == 5:
        run_fn(out0, out1, out2, in0, in1, in2, in3, in4)
    elif num_outs == 3 and num_ins == 6:
        run_fn(out0, out1, out2, in0, in1, in2, in3, in4, in5)
    elif num_outs == 4 and num_ins == 1:
        run_fn(out0, out1, out2, out3, in0)
    elif num_outs == 4 and num_ins == 2:
        run_fn(out0, out1, out2, out3, in0, in1)
    elif num_outs == 4 and num_ins == 3:
        run_fn(out0, out1, out2, out3, in0, in1, in2)
    elif num_outs == 4 and num_ins == 4:
        run_fn(out0, out1, out2, out3, in0, in1, in2, in3)
    elif num_outs == 4 and num_ins == 5:
        run_fn(out0, out1, out2, out3, in0, in1, in2, in3, in4)
    elif num_outs == 4 and num_ins == 6:
        run_fn(out0, out1, out2, out3, in0, in1, in2, in3, in4, in5)
    elif num_outs == 5 and num_ins == 1:
        run_fn(out0, out1, out2, out3, out4, in0)
    elif num_outs == 5 and num_ins == 2:
        run_fn(out0, out1, out2, out3, out4, in0, in1)
    elif num_outs == 5 and num_ins == 3:
        run_fn(out0, out1, out2, out3, out4, in0, in1, in2)
    elif num_outs == 5 and num_ins == 4:
        run_fn(out0, out1, out2, out3, out4, in0, in1, in2, in3)
    elif num_outs == 5 and num_ins == 5:
        run_fn(out0, out1, out2, out3, out4, in0, in1, in2, in3, in4)
    elif num_outs == 5 and num_ins == 6:
        run_fn(out0, out1, out2, out3, out4, in0, in1, in2, in3, in4, in5)
    elif num_outs == 6 and num_ins == 1:
        run_fn(out0, out1, out2, out3, out4, out5, in0)
    elif num_outs == 6 and num_ins == 2:
        run_fn(out0, out1, out2, out3, out4, out5, in0, in1)
    elif num_outs == 6 and num_ins == 3:
        run_fn(out0, out1, out2, out3, out4, out5, in0, in1, in2)
    elif num_outs == 6 and num_ins == 4:
        run_fn(out0, out1, out2, out3, out4, out5, in0, in1, in2, in3)
    elif num_outs == 6 and num_ins == 5:
        run_fn(out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4)
    else:
        run_fn(out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5)

ops._wrap_op(NumbaFunc, "fn", __name__)