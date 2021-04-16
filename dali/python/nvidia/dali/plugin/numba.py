from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops
from nvidia.dali import types as dali_types
from numba import types as numba_types
from numba import njit, cfunc, carray
import numpy as np
import numba as nb

_to_numpy = {
    dali_types.UINT8 : "uint8",
    dali_types.UINT16 : "uint16",
    dali_types.UINT32 : "uint32",
    dali_types.UINT64 : "uint64",
    dali_types.INT8 : "int8",
    dali_types.INT16 : "int16",
    dali_types.INT32 : "int32",
    dali_types.INT64 : "int64",
    dali_types.FLOAT16 : "float16",
    dali_types.FLOAT : "float32",
    dali_types.FLOAT64 : "float64",
}

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

class NumbaFunc(ops.NumbaFunctionBase):
    ops.register_cpu_op('NumbaFunc')

    def _setup_fn_sig(self):
        return numba_types.void(numba_types.uint64,
                        numba_types.uint64,
                        numba_types.int32,
                        numba_types.uint64,
                        numba_types.uint64,
                        numba_types.int32, numba_types.int32)
                
    def _run_fn_sig(self, batch_processing=False):
        sig_types = []
        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.int32)

        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.uint64)
        sig_types.append(numba_types.int32)

        if batch_processing:
            sig_types.append(numba_types.int32)
        return numba_types.void(*sig_types)

    def _get_carray_eval_lambda(self, dtype, ndim):
        eval_string = "lambda ptr, shape: carray(ptr, ("
        for i in range(ndim):
            eval_string += "shape[{}]".format(i)
            eval_string += ", " if i + 1 != ndim else "), "
        eval_string += "dtype=np.{})".format(_to_numpy[dtype])
        return njit(eval(eval_string))

    def _get_carrays_eval_lambda(self, types, ndim):
        return tuple([self._get_carray_eval_lambda(dtype, ndim) for dtype, ndim in zip(types, ndim)] + [njit(eval(("lambda x, y: None"))) for i in range(6 - len(types))])

    def _get_run_fn_lambda(self, num_outs, num_ins):
        eval_string = "lambda run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5 : "
        eval_string += "run_fn("
        for i in range(num_outs):
            eval_string += "out{}".format(i)
            eval_string += ", " if i + 1 != num_outs else  ", "
        for i in range(num_ins):
            eval_string += "in{}".format(i)
            eval_string += ", " if i + 1 != num_ins else  ")"
        return njit(eval(eval_string))

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        if pipeline is None:
            Pipeline._raise_no_current_pipeline("NumbaFunc")
        return super(NumbaFunc, self).__call__(*inputs, **kwargs)

    def __init__(self, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn=None, device='cpu', batch_processing=False, **kwargs):
        setup_fn_address = None
        if setup_fn != None:
            setup_fn = njit(setup_fn)
            @cfunc(self._setup_fn_sig(), nopython=True)
            def setup_cfunc(out_shapes_ptr, out_ndims_ptr, num_outs, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
                out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, num_samples)
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_outs, num_samples)
                setup_fn(out_shapes_np, in_shapes_np)
            setup_fn_address = setup_cfunc.address

        out0_lambda, out1_lambda, out2_lambda, out3_lambda, out4_lambda, out5_lambda = self._get_carrays_eval_lambda(out_types, outs_ndim)
        in0_lambda, in1_lambda, in2_lambda, in3_lambda, in4_lambda, in5_lambda = self._get_carrays_eval_lambda(in_types, ins_ndim)
        run_fn = njit(run_fn)
        run_fn_lambda = self._get_run_fn_lambda(len(out_types), len(in_types))
        if batch_processing:
            @cfunc(self._run_fn_sig(True), nopython=True)
            def run_cfunc(out_ptr, out_shapes_ptr, out_ndims_ptr, num_outs, in_ptr, in_shapes_ptr, in_ndims_ptr, num_ins, num_samples):
                out0 = out1 = out2 = out3 = out4 = out5 = None
                out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, num_samples)
                out_arr = carray(address_as_void_pointer(out_ptr), (num_outs, num_samples), dtype=np.int64)
                if num_outs >= 1:
                    out0 = [out0_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[0], out_shapes_np[0])]
                if num_outs >= 2:
                    out1 = [out1_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[1], out_shapes_np[1])]
                if num_outs >= 3:
                    out2 = [out2_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[2], out_shapes_np[2])]
                if num_outs >= 4:
                    out3 = [out3_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[3], out_shapes_np[3])]
                if num_outs >= 5:
                    out4 = [out4_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[4], out_shapes_np[4])]
                if num_outs >= 6:
                    out5 = [out5_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(out_arr[5], out_shapes_np[5])]

                in0 = in1 = in2 = in3 = in4 = in5 = None
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_ins, num_samples)
                in_arr = carray(address_as_void_pointer(in_ptr), (num_ins, num_samples), dtype=np.int64)
                if num_ins >= 1:
                    in0 = [in0_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[0], in_shapes_np[0])]
                if num_ins >= 2:
                    in1 = [in1_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[1], in_shapes_np[1])]
                if num_ins >= 3:
                    in2 = [in2_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[2], in_shapes_np[2])]
                if num_ins >= 4:
                    in3 = [in3_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[3], in_shapes_np[3])]
                if num_ins >= 5:
                    in4 = [in4_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[4], in_shapes_np[4])]
                if num_ins >= 6:
                    in5 = [in5_lambda(address_as_void_pointer(ptr), shape) for ptr, shape in zip(in_arr[5], in_shapes_np[5])]

                run_fn_lambda(run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5)
        else:
            @cfunc(self._run_fn_sig(), nopython=True)
            def run_cfunc(out_ptr, out_shapes_ptr, out_ndims_ptr, num_outs, in_ptr, in_shapes_ptr, in_ndims_ptr, num_ins):
                out0 = out1 = out2 = out3 = out4 = out5 = None
                out_shapes_np = _get_shape_view(out_shapes_ptr, out_ndims_ptr, num_outs, 1)
                out_arr = carray(address_as_void_pointer(out_ptr), num_outs, dtype=np.int64)
                if num_outs >= 1:
                    out0 = out0_lambda(address_as_void_pointer(out_arr[0]), out_shapes_np[0][0])
                if num_outs >= 2:
                    out1 = out1_lambda(address_as_void_pointer(out_arr[1]), out_shapes_np[1][0])
                if num_outs >= 3:
                    out2 = out2_lambda(address_as_void_pointer(out_arr[2]), out_shapes_np[2][0])
                if num_outs >= 4:
                    out3 = out3_lambda(address_as_void_pointer(out_arr[3]), out_shapes_np[3][0])
                if num_outs >= 5:
                    out4 = out4_lambda(address_as_void_pointer(out_arr[4]), out_shapes_np[4][0])
                if num_outs >= 6:
                    out5 = out5_lambda(address_as_void_pointer(out_arr[5]), out_shapes_np[5][0])
                
                in0 = in1 = in2 = in3 = in4 = in5 = None
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_ins, 1)
                in_arr = carray(address_as_void_pointer(in_ptr), num_ins, dtype=np.int64)
                if num_ins >= 1:
                    in0 = in0_lambda(address_as_void_pointer(in_arr[0]), in_shapes_np[0][0])
                if num_ins >= 2:
                    in1 = in1_lambda(address_as_void_pointer(in_arr[1]), in_shapes_np[1][0])
                if num_ins >= 3:
                    in2 = in2_lambda(address_as_void_pointer(in_arr[2]), in_shapes_np[2][0])
                if num_ins >= 4:
                    in3 = in3_lambda(address_as_void_pointer(in_arr[3]), in_shapes_np[3][0])
                if num_ins >= 5:
                    in4 = in4_lambda(address_as_void_pointer(in_arr[4]), in_shapes_np[4][0])
                if num_ins >= 6:
                    in5 = in5_lambda(address_as_void_pointer(in_arr[5]), in_shapes_np[5][0])

                run_fn_lambda(run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5)

        super(NumbaFunc, self).__init__(impl_name="NumbaFuncImpl",
                                                setup_fn=setup_fn_address, run_fn=run_cfunc.address,
                                                out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim,
                                                batch_processing=batch_processing, device=device, **kwargs)


ops._wrap_op(NumbaFunc, "fn", __name__)
