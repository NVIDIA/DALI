# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import nvidia.dali.python_function_plugin
from nvidia.dali import ops
from nvidia.dali.ops import _registry
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.types import CUDAStream as _CUDAStream


cupy = None


def _setup_cupy():
    global cupy
    if cupy is None:
        import cupy as cupy


def _get_base_impl(name, impl_name):

    class PythonFunctionBase(ops.python_op_factory(impl_name, name, impl_name, generated=False)):

        def __init__(self, function, num_outputs=1, **kwargs):

            # The layouts need to be handled manually due to an implementation detail
            # By calling spec.AddArg manually, we skip the promotion from a single string argument
            # to a 1-element list of strings that is done by the automation in the base class.
            # This way, the operator is able to differentiate between those cases.
            self._output_layouts = kwargs.pop("output_layouts", None)
            super().__init__(**kwargs)
            if self._output_layouts is not None:
                self._spec.AddArg("output_layouts", self._output_layouts)

            self.function = function
            self.num_outputs = num_outputs
            self._preserve = True

        def __call__(self, *inputs, **kwargs):
            inputs = ops._preprocess_inputs(inputs, impl_name, self._device, None)
            curr_pipe = _Pipeline.current()
            if curr_pipe is None:
                _Pipeline._raise_pipeline_required("PythonFunction operator")
            self.pipeline = curr_pipe._stub()

            for inp in inputs:
                if not isinstance(inp, _DataNode):
                    raise TypeError(
                        f"Expected inputs of type `DataNode`. "
                        f"Received input of type '{type(inp).__name__}'. "
                        f"Python Operators do not support Multiple Input Sets."
                    )

            call_layouts = kwargs.pop("output_layouts", None)
            if self._output_layouts is not None:
                # For the purpose of erroring on double definition
                ops._resolve_double_definitions(
                    {"output_layouts": call_layouts}, {"output_layouts": self._output_layouts}
                )
            elif call_layouts is not None:
                self._spec.AddArg("output_layouts", call_layouts)

            kwargs.update({"function_id": id(self.function), "num_outputs": self.num_outputs})

            return super().__call__(*inputs, **kwargs)

    return PythonFunctionBase


def _dlpack_to_array(dlpack):
    return nvidia.dali.python_function_plugin.DLTensorToArray(dlpack)


def _dlpack_from_array(array):
    return nvidia.dali.python_function_plugin.ArrayToDLTensor(array)


class PythonFunction(_get_base_impl("PythonFunction", "DLTensorPythonFunctionImpl")):
    _registry.register_cpu_op("PythonFunction")
    _registry.register_gpu_op("PythonFunction")

    @staticmethod
    def current_stream():
        """Gets DALI's current CUDA stream."""
        return _CUDAStream(nvidia.dali.python_function_plugin.current_dali_stream())

    @staticmethod
    def check_outputs(outputs, num_outputs):
        if num_outputs > 1:
            if not isinstance(outputs, tuple):
                raise TypeError(
                    "The output from a multi-output Python"
                    "function operator must be a tuple, got: ",
                    type(outputs),
                )
            if len(outputs) != num_outputs:
                raise ValueError(
                    f"Unexpected number of outputs from Python"
                    f"function operator - got {len(outputs)}, expected {num_outputs}"
                )

    @staticmethod
    def function_wrapper_per_sample(
        pipeline, function, num_outputs, from_dlpack, to_dlpack, *dlpack_inputs
    ):
        with pipeline:
            arrays = [from_dlpack(dlpack) for dlpack in dlpack_inputs]
            arr_out = function(*arrays)
            if arr_out is None:
                return
            PythonFunction.check_outputs(arr_out, num_outputs)
            if isinstance(arr_out, tuple):
                return tuple(map(lambda t: to_dlpack(t), arr_out))
            else:
                return to_dlpack(arr_out)

    @staticmethod
    def function_wrapper_batch(
        pipeline, function, num_outputs, from_dlpack, to_dlpack, *dlpack_inputs
    ):
        with pipeline:
            arrays = [[from_dlpack(dlpack) for dlpack in dl_input] for dl_input in dlpack_inputs]
            arr_outs = function(*arrays)
            if arr_outs is None:
                return

            def convert_batch(batch):
                if isinstance(batch, list):
                    return [to_dlpack(x) for x in batch]
                else:
                    return to_dlpack(batch)

            PythonFunction.check_outputs(arr_outs, num_outputs)
            if isinstance(arr_outs, tuple):
                return tuple(convert_batch(x) for x in arr_outs)
            else:
                return convert_batch(arr_outs)

    def _function_wrapper_cpu(self, batch_processing, function, num_outputs, *dlpack_inputs):
        if batch_processing:
            return PythonFunction.function_wrapper_batch(
                self.pipeline,
                function,
                num_outputs,
                _dlpack_to_array,
                _dlpack_from_array,
                *dlpack_inputs,
            )
        else:
            return PythonFunction.function_wrapper_per_sample(
                self.pipeline,
                function,
                num_outputs,
                _dlpack_to_array,
                _dlpack_from_array,
                *dlpack_inputs,
            )

    @staticmethod
    def _cupy_stream_wrapper(function, *inputs):
        stream = cupy.cuda.Stream(null=True)
        stream.ptr = PythonFunction.current_stream().ptr
        with stream:
            out = function(*inputs)
        stream.ptr = 0
        return out

    def _function_wrapper_gpu(self, batch_processing, function, num_outputs, *dlpack_inputs):
        def wrapped_func(*inputs):
            return PythonFunction._cupy_stream_wrapper(function, *inputs)

        if batch_processing:
            return PythonFunction.function_wrapper_batch(
                self.pipeline,
                wrapped_func,
                num_outputs,
                cupy.fromDlpack,
                lambda t: t.toDlpack(),
                *dlpack_inputs,
            )
        else:
            return PythonFunction.function_wrapper_per_sample(
                self.pipeline,
                wrapped_func,
                num_outputs,
                cupy.fromDlpack,
                lambda t: t.toDlpack(),
                *dlpack_inputs,
            )

    def __init__(self, function, num_outputs=1, device="cpu", batch_processing=False, **kwargs):
        if device == "gpu":
            _setup_cupy()

        if device == "cpu":

            def func(*ts):
                return self._function_wrapper_cpu(batch_processing, function, num_outputs, *ts)

        else:

            def func(*ts):
                return self._function_wrapper_gpu(batch_processing, function, num_outputs, *ts)

        super().__init__(
            function=func,
            num_outputs=num_outputs,
            device=device,
            synchronize_stream=False,
            batch_processing=batch_processing,
            **kwargs,
        )


class DLTensorPythonFunction(
    _get_base_impl("DLTensorPythonFunction", "DLTensorPythonFunctionImpl")
):
    _registry.register_cpu_op("DLTensorPythonFunction")
    _registry.register_gpu_op("DLTensorPythonFunction")

    def _function_wrapper_dlpack(self, batch_processing, function, num_outputs, *dlpack_inputs):
        if batch_processing:
            return PythonFunction.function_wrapper_batch(
                self.pipeline, function, num_outputs, lambda x: x, lambda x: x, *dlpack_inputs
            )
        else:
            return PythonFunction.function_wrapper_per_sample(
                self.pipeline, function, num_outputs, lambda x: x, lambda x: x, *dlpack_inputs
            )

    def __init__(
        self,
        function,
        num_outputs=1,
        device="cpu",
        synchronize_stream=True,
        batch_processing=True,
        **kwargs,
    ):
        def func(*ts):
            return self._function_wrapper_dlpack(batch_processing, function, num_outputs, *ts)

        super().__init__(
            function=func,
            num_outputs=num_outputs,
            device=device,
            synchronize_stream=synchronize_stream,
            batch_processing=batch_processing,
            **kwargs,
        )
