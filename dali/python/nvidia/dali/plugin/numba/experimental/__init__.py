# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from packaging.version import Version

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import ops
from nvidia.dali import types as dali_types
from numba import types as numba_types
from numba import njit, cfunc, carray
from numba import cuda as nb_cuda
import numpy as np
import numba as nb
import importlib


_to_numpy = {
    dali_types.BOOL: "bool_",
    dali_types.UINT8: "uint8",
    dali_types.UINT16: "uint16",
    dali_types.UINT32: "uint32",
    dali_types.UINT64: "uint64",
    dali_types.INT8: "int8",
    dali_types.INT16: "int16",
    dali_types.INT32: "int32",
    dali_types.INT64: "int64",
    dali_types.FLOAT16: "float16",
    dali_types.FLOAT: "float32",
    dali_types.FLOAT64: "float64",
}

_to_numba = {
    dali_types.BOOL: numba_types.boolean,
    dali_types.UINT8: numba_types.uint8,
    dali_types.UINT16: numba_types.uint16,
    dali_types.UINT32: numba_types.uint32,
    dali_types.UINT64: numba_types.uint64,
    dali_types.INT8: numba_types.int8,
    dali_types.INT16: numba_types.int16,
    dali_types.INT32: numba_types.int32,
    dali_types.INT64: numba_types.int64,
    dali_types.FLOAT16: numba_types.float16,
    dali_types.FLOAT: numba_types.float32,
    dali_types.FLOAT64: numba_types.float64,
}


# Minimal version of Numba that is required for Numba GPU operator to work
minimal_numba_version = {
    11: Version("0.55.2"),
    12: Version("0.57.0"),
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
    ret = []
    for sample, size in zip(samples, ndims):
        d = []
        for shape_ptr in sample:
            d.append(carray(address_as_void_pointer(shape_ptr), size, dtype=np.int64))
        ret.append(d)
    return ret


class NumbaFunction(
    ops.python_op_factory("NumbaFunctionBase", "NumbaFunction", "NumbaFuncImpl", generated=False)
):
    _impl_module = "nvidia.dali.plugin.numba"
    ops.register_cpu_op("NumbaFunction")
    ops.register_gpu_op("NumbaFunction")

    @property
    def spec(self):
        return self._spec

    @property
    def schema(self):
        return self._schema

    @property
    def device(self):
        return self._device

    @property
    def preserve(self):
        return self._preserve

    def _setup_fn_sig(self):
        return numba_types.void(
            numba_types.uint64,
            numba_types.uint64,
            numba_types.int32,
            numba_types.uint64,
            numba_types.uint64,
            numba_types.int32,
            numba_types.int32,
        )

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
        return njit(eval(eval_string))  # nosec B307

    def _get_carrays_eval_lambda(self, types, ndim):
        ret = [self._get_carray_eval_lambda(dtype, ndim) for dtype, ndim in zip(types, ndim)]
        ret += [njit(eval(("lambda x, y: None"))) for i in range(6 - len(types))]  # nosec B307
        return tuple(ret)

    def _get_run_fn_lambda(self, num_outs, num_ins):
        eval_string = (
            "lambda run_fn, out0, out1, out2, out3, out4, out5, "
            "in0, in1, in2, in3, in4, in5 : "
            "run_fn("
        )
        for i in range(num_outs):
            eval_string += "out{}".format(i)
            eval_string += ", " if i + 1 != num_outs else ", "
        for i in range(num_ins):
            eval_string += "in{}".format(i)
            eval_string += ", " if i + 1 != num_ins else ")"
        return njit(eval(eval_string))  # nosec B307

    def _get_setup_fn_cpu(self, setup_fn):
        setup_fn_address = None
        if setup_fn is not None:
            setup_fn = njit(setup_fn)

            @cfunc(self._setup_fn_sig(), nopython=True)
            def setup_cfunc(
                out_shapes_ptr,
                out_ndims_ptr,
                num_outs,
                in_shapes_ptr,
                in_ndims_ptr,
                num_ins,
                num_samples,
            ):
                out_shapes_np = _get_shape_view(
                    out_shapes_ptr, out_ndims_ptr, num_outs, num_samples
                )
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_outs, num_samples)
                setup_fn(out_shapes_np, in_shapes_np)

            setup_fn_address = setup_cfunc.address

        return setup_fn_address

    def _get_run_fn_gpu(self, run_fn, types, dims):
        nvvm_options = {"fastmath": False, "opt": 3}

        cuda_arguments = []
        for dali_type, ndim in zip(types, dims):
            cuda_arguments.append(numba_types.Array(_to_numba[dali_type], ndim, "C"))

        if Version(nb.__version__) < Version("0.57.0"):
            cres = nb_cuda.compiler.compile_cuda(run_fn, numba_types.void, cuda_arguments)
        else:
            pipeline = Pipeline.current()
            device_id = pipeline.device_id
            old_device = nb_cuda.api.get_current_device().id
            cc = nb_cuda.api.select_device(device_id).compute_capability
            nb_cuda.api.select_device(old_device)
            cres = nb_cuda.compiler.compile_cuda(
                run_fn,
                numba_types.void,
                cuda_arguments,
                nvvm_options=nvvm_options,
                fastmath=False,
                cc=cc,
            )

        tgt_ctx = cres.target_context
        code = run_fn.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno
        return_value = 0
        if Version(nb.__version__) < Version("0.57.0"):
            nvvm_options["debug"] = False
            nvvm_options["lineinfo"] = False
            lib, _ = tgt_ctx.prepare_cuda_kernel(
                cres.library, cres.fndesc, True, nvvm_options, filename, linenum
            )
            return_value = lib.get_cufunc().handle.value
        else:
            if hasattr(tgt_ctx, "prepare_cuda_kernel"):
                lib, _ = tgt_ctx.prepare_cuda_kernel(
                    cres.library, cres.fndesc, False, True, nvvm_options, filename, linenum
                )
                return_value = lib.get_cufunc().handle.value
            else:
                from numba.cuda.compiler import kernel_fixup

                lib = cres.library
                kernel = lib.get_function(cres.fndesc.llvm_func_name)
                kernel_fixup(kernel, debug=False)
                lib._entry_name = cres.fndesc.llvm_func_name
                return_value = int(lib.get_cufunc().handle)

        return return_value

    def _get_run_fn_cpu(self, run_fn, out_types, in_types, outs_ndim, ins_ndim, batch_processing):
        (
            out0_lambda,
            out1_lambda,
            out2_lambda,
            out3_lambda,
            out4_lambda,
            out5_lambda,
        ) = self._get_carrays_eval_lambda(out_types, outs_ndim)
        (
            in0_lambda,
            in1_lambda,
            in2_lambda,
            in3_lambda,
            in4_lambda,
            in5_lambda,
        ) = self._get_carrays_eval_lambda(in_types, ins_ndim)
        run_fn = njit(run_fn)
        run_fn_lambda = self._get_run_fn_lambda(len(out_types), len(in_types))
        if batch_processing:

            @cfunc(self._run_fn_sig(batch_processing=True), nopython=True)
            def run_cfunc(
                out_ptr,
                out_shapes_ptr,
                out_ndims_ptr,
                num_outs,
                in_ptr,
                in_shapes_ptr,
                in_ndims_ptr,
                num_ins,
                num_samples,
            ):
                out0 = out1 = out2 = out3 = out4 = out5 = None
                out_shapes_np = _get_shape_view(
                    out_shapes_ptr, out_ndims_ptr, num_outs, num_samples
                )
                out_arr = carray(
                    address_as_void_pointer(out_ptr), (num_outs, num_samples), dtype=np.int64
                )
                if num_outs >= 1:
                    out0 = [
                        out0_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[0], out_shapes_np[0])
                    ]
                if num_outs >= 2:
                    out1 = [
                        out1_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[1], out_shapes_np[1])
                    ]
                if num_outs >= 3:
                    out2 = [
                        out2_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[2], out_shapes_np[2])
                    ]
                if num_outs >= 4:
                    out3 = [
                        out3_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[3], out_shapes_np[3])
                    ]
                if num_outs >= 5:
                    out4 = [
                        out4_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[4], out_shapes_np[4])
                    ]
                if num_outs >= 6:
                    out5 = [
                        out5_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(out_arr[5], out_shapes_np[5])
                    ]

                in0 = in1 = in2 = in3 = in4 = in5 = None
                in_shapes_np = _get_shape_view(in_shapes_ptr, in_ndims_ptr, num_ins, num_samples)
                in_arr = carray(
                    address_as_void_pointer(in_ptr), (num_ins, num_samples), dtype=np.int64
                )
                if num_ins >= 1:
                    in0 = [
                        in0_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[0], in_shapes_np[0])
                    ]
                if num_ins >= 2:
                    in1 = [
                        in1_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[1], in_shapes_np[1])
                    ]
                if num_ins >= 3:
                    in2 = [
                        in2_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[2], in_shapes_np[2])
                    ]
                if num_ins >= 4:
                    in3 = [
                        in3_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[3], in_shapes_np[3])
                    ]
                if num_ins >= 5:
                    in4 = [
                        in4_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[4], in_shapes_np[4])
                    ]
                if num_ins >= 6:
                    in5 = [
                        in5_lambda(address_as_void_pointer(ptr), shape)
                        for ptr, shape in zip(in_arr[5], in_shapes_np[5])
                    ]

                run_fn_lambda(
                    run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5
                )

        else:

            @cfunc(self._run_fn_sig(batch_processing=False), nopython=True)
            def run_cfunc(
                out_ptr,
                out_shapes_ptr,
                out_ndims_ptr,
                num_outs,
                in_ptr,
                in_shapes_ptr,
                in_ndims_ptr,
                num_ins,
            ):
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

                run_fn_lambda(
                    run_fn, out0, out1, out2, out3, out4, out5, in0, in1, in2, in3, in4, in5
                )

        return run_cfunc.address

    def __call__(self, *inputs, **kwargs):
        pipeline = Pipeline.current()
        inputs = ops._preprocess_inputs(inputs, self.__class__.__name__, self._device, None)
        if pipeline is None:
            Pipeline._raise_pipeline_required(self.__class__.__name__)
        for inp in inputs:
            if not isinstance(inp, _DataNode):
                raise TypeError(
                    (
                        "Expected inputs of type `DataNode`. Received input of type '{}'. "
                        + "Python Operators do not support Multiple Input Sets."
                    ).format(type(inp).__name__)
                )

        kwargs.update(
            {
                "run_fn": self.run_fn,
                "out_types": self.out_types,
                "in_types": self.in_types,
                "outs_ndim": self.outs_ndim,
                "ins_ndim": self.ins_ndim,
                "batch_processing": self.batch_processing,
            }
        )
        if self.setup_fn is not None:
            kwargs.update({"setup_fn": self.setup_fn})
        if self.device == "gpu":
            kwargs.update(
                {
                    "blocks": self.blocks,
                    "threads_per_block": self.threads_per_block,
                }
            )

        return super().__call__(*inputs, **kwargs)

    def __init__(
        self,
        run_fn,
        out_types,
        in_types,
        outs_ndim,
        ins_ndim,
        setup_fn=None,
        device="cpu",
        batch_processing=False,
        blocks=None,
        threads_per_block=None,
        **kwargs,
    ):
        if device == "gpu":
            NumbaFunction._check_minimal_numba_version()
            NumbaFunction._check_cuda_compatibility()

        # TODO(klecki): Normalize the types into lists first, than apply the checks
        assert len(in_types) == len(ins_ndim), (
            "Number of input types " "and input dimensions should match."
        )
        assert len(out_types) == len(outs_ndim), (
            "Number of output types " "and output dimensions should match."
        )

        if "float16" in dir(numba_types):
            for t in [*in_types, *out_types]:
                if t == dali_types.FLOAT16:
                    raise RuntimeError(
                        "Numba does not support float16 for "
                        "current Python version. "
                        "Python 3.7 or newer is required"
                    )

        if device == "gpu":
            assert batch_processing is False, "Currently batch processing for GPU is not supported."
            assert len(blocks) == 3, (
                "`blocks` array should contain 3 numbers, while received: " f"{len(blocks)}"
            )
            for i, block_dim in enumerate(blocks):
                assert block_dim > 0, (
                    "All dimensions should be positive. Value specified in "
                    f"`blocks` at index {i} is nonpositive: {block_dim}"
                )

            assert len(threads_per_block) == 3, (
                "`threads_per_block` array should contain 3 "
                f"numbers, while received: {len(threads_per_block)}"
            )
            for i, threads in enumerate(threads_per_block):
                assert threads > 0, (
                    "All dimensions should be positive. "
                    "Value specified in `threads_per_block` at index "
                    f"{i} is nonpositive: {threads}"
                )

        if not isinstance(outs_ndim, list):
            outs_ndim = [outs_ndim]
        if not isinstance(ins_ndim, list):
            ins_ndim = [ins_ndim]
        if not isinstance(out_types, list):
            out_types = [out_types]
        if not isinstance(in_types, list):
            in_types = [in_types]

        super().__init__(device=device, **kwargs)

        if device == "gpu":
            self.run_fn = self._get_run_fn_gpu(run_fn, out_types + in_types, outs_ndim + ins_ndim)
        else:
            self.run_fn = self._get_run_fn_cpu(
                run_fn, out_types, in_types, outs_ndim, ins_ndim, batch_processing
            )
        self.setup_fn = self._get_setup_fn_cpu(setup_fn)
        self.out_types = out_types
        self.in_types = in_types
        self.outs_ndim = outs_ndim
        self.ins_ndim = ins_ndim
        self.num_outputs = len(out_types)
        self.batch_processing = batch_processing
        self._preserve = True
        self.blocks = blocks
        self.threads_per_block = threads_per_block

    @staticmethod
    def _check_minimal_numba_version(throw: bool = True):
        current_version = Version(nb.__version__)
        toolkit_version = nb_cuda.runtime.get_version()
        if toolkit_version[0] not in minimal_numba_version:
            if throw:
                raise RuntimeError(f"Unsupported CUDA toolkit version: {toolkit_version}")
            else:
                return False
        min_ver = minimal_numba_version[toolkit_version[0]]
        if current_version < min_ver:
            if throw:
                raise RuntimeError(
                    f"Insufficient Numba version. Numba GPU operator "
                    f"requires Numba {str(min_ver)} or higher. "
                    f"Detected version: {str(Version(nb.__version__))}."
                )
            else:
                return False
        return True

    @staticmethod
    def _check_cuda_compatibility(throw: bool = True):
        toolkit_version = nb_cuda.runtime.get_version()
        driver_version = nb_cuda.driver.driver.get_version()

        # numba_cuda should handle the compatibility between toolkit and driver versions
        # otherwise check if the driver and runtime matches, or if the last working numba version
        # matches the driver for CUDA 12
        try:
            # try importing cuda.core as it can be used later to check the compatibility
            # it is okay to fail as it may not be installed, the check later can handle this
            import cuda.core
        except ImportError:
            pass

        # numba_cuda similarly to numba provides numba.cuda module so we need
        # to check is package is present to learn who provides it
        numba_cuda_missing = not importlib.util.find_spec("numba_cuda")
        cuda_core_too_old = (
            importlib.util.find_spec("core")
            and importlib.util.find_spec("cuda.core")
            and Version(cuda.core.__version__) <= Version("0.3.1")
            and nb_cuda.driver.driver.get_version()[0] > 12
        )
        toolkit_newer_than_driver = toolkit_version > driver_version
        numba_too_old_for_driver = (
            Version(nb.__version__) <= Version("0.61.2")
            and nb_cuda.driver.driver.get_version()[0] > 12
        )

        if numba_cuda_missing or cuda_core_too_old:
            if toolkit_newer_than_driver or numba_too_old_for_driver:
                if throw:
                    raise RuntimeError(
                        f"Environment is not compatible with Numba GPU operator. "
                        f"Driver version is {driver_version} and CUDA Toolkit "
                        f"version is {toolkit_version}. "
                        "Driver cannot be older than the CUDA Toolkit"
                    )
                else:
                    return False
        return True


ops._wrap_op(NumbaFunction, "fn.experimental", "nvidia.dali.plugin.numba")
