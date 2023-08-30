# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import backend as _b
from nvidia.dali import _conditionals
from nvidia.dali import ops
from nvidia.dali.ops import _registry
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.types import (CUDAStream as _CUDAStream)


cupy = None


def _setup_cupy():
    global cupy
    if cupy is None:
        import cupy as cupy


class PythonFunctionBase(metaclass=ops._DaliOperatorMeta):

    def __init__(self, impl_name, function, num_outputs=1, device='cpu', **kwargs):
        self._schema = _b.GetSchema(impl_name)
        self._spec = _b.OpSpec(impl_name)
        self._device = device
        self._impl_name = impl_name

        kwargs, self._call_args = ops._separate_kwargs(kwargs)

        for key, value in kwargs.items():
            self._spec.AddArg(key, value)

        self.function = function
        self.num_outputs = num_outputs
        self._preserve = True

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

    def __call__(self, *inputs, **kwargs):
        inputs = ops._preprocess_inputs(inputs, self._impl_name, self._device, None)
        pipeline = _Pipeline.current()
        if pipeline is None:
            _Pipeline._raise_pipeline_required("PythonFunction operator")

        if (len(inputs) > self._schema.MaxNumInput() or len(inputs) < self._schema.MinNumInput()):
            raise ValueError(
                f"Operator {type(self).__name__} expects "
                f"from {self._schema.MinNumInput()} to {self._schema.MaxNumInput()} inputs, "
                f"but received {len(inputs)}.")
        for inp in inputs:
            if not isinstance(inp, _DataNode):
                raise TypeError(f"Expected inputs of type `DataNode`. "
                                f"Received input of type '{type(inp).__name__}'. "
                                f"Python Operators do not support Multiple Input Sets.")
        op_instance = ops._OperatorInstance(inputs, self, **kwargs)
        op_instance.spec.AddArg("function_id", id(self.function))
        op_instance.spec.AddArg("num_outputs", self.num_outputs)
        op_instance.spec.AddArg("device", self.device)
        if self.num_outputs == 0:
            t_name = self._impl_name + "_id_" + str(op_instance.id) + "_sink"
            t = _DataNode(t_name, self._device, op_instance)
            pipeline.add_sink(t)
            return
        outputs = []

        for i in range(self.num_outputs):
            t_name = op_instance._name
            if self.num_outputs > 1:
                t_name += "[{}]".format(i)
            t = _DataNode(t_name, self._device, op_instance)
            op_instance.spec.AddOutput(t.name, t.device)
            op_instance.append_output(t)
            pipeline.add_sink(t)
            outputs.append(t)

        if _conditionals.conditionals_enabled():
            _conditionals.register_data_nodes(outputs, inputs, kwargs)
        return outputs[0] if len(outputs) == 1 else outputs


def _dlpack_to_array(dlpack):
    return nvidia.dali.python_function_plugin.DLTensorToArray(dlpack)


def _dlpack_from_array(array):
    return nvidia.dali.python_function_plugin.ArrayToDLTensor(array)


class PythonFunction(PythonFunctionBase):
    schema_name = "PythonFunction"
    _registry.register_cpu_op('PythonFunction')
    _registry.register_gpu_op('PythonFunction')

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
                    "function operator must be a tuple, got: ", type(outputs))
            if len(outputs) != num_outputs:
                raise ValueError(f"Unexpected number of outputs from Python"
                                 f"function operator - got {len(outputs)}, expected {num_outputs}")

    @staticmethod
    def function_wrapper_per_sample(function, num_outputs, from_dlpack, to_dlpack, *dlpack_inputs):
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
    def function_wrapper_batch(function, num_outputs, from_dlpack, to_dlpack, *dlpack_inputs):
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

    @staticmethod
    def _function_wrapper_cpu(batch_processing, function, num_outputs, *dlpack_inputs):
        if batch_processing:
            return PythonFunction.function_wrapper_batch(
                function,
                num_outputs,
                _dlpack_to_array,
                _dlpack_from_array,
                *dlpack_inputs)
        else:
            return PythonFunction.function_wrapper_per_sample(
                function,
                num_outputs,
                _dlpack_to_array,
                _dlpack_from_array,
                *dlpack_inputs)

    @staticmethod
    def _cupy_stream_wrapper(function, *inputs):
        stream = cupy.cuda.Stream(null=True)
        stream.ptr = PythonFunction.current_stream().ptr
        with stream:
            out = function(*inputs)
        stream.ptr = 0
        return out

    @staticmethod
    def _function_wrapper_gpu(batch_processing, function, num_outputs, *dlpack_inputs):

        def wrapped_func(*inputs):
            return PythonFunction._cupy_stream_wrapper(function, *inputs)

        if batch_processing:
            return PythonFunction.function_wrapper_batch(wrapped_func, num_outputs, cupy.fromDlpack,
                                                         lambda t: t.toDlpack(), *dlpack_inputs)
        else:
            return PythonFunction.function_wrapper_per_sample(wrapped_func, num_outputs,
                                                              cupy.fromDlpack,
                                                              lambda t: t.toDlpack(),
                                                              *dlpack_inputs)

    def __init__(self, function, num_outputs=1, device='cpu', batch_processing=False, **kwargs):
        if device == 'gpu':
            _setup_cupy()

        if device == 'cpu':
            def func(*ts):
                return PythonFunction._function_wrapper_cpu(
                    batch_processing, function, num_outputs, *ts)
        else:
            def func(*ts):
                return PythonFunction._function_wrapper_gpu(
                    batch_processing, function, num_outputs, *ts)

        super(PythonFunction,
              self).__init__(
                impl_name="DLTensorPythonFunctionImpl",
                function=func,
                num_outputs=num_outputs,
                device=device,
                synchronize_stream=False,
                batch_processing=batch_processing,
                **kwargs)


class DLTensorPythonFunction(PythonFunctionBase):
    schema_name = "DLTensorPythonFunction"
    _registry.register_cpu_op('DLTensorPythonFunction')
    _registry.register_gpu_op('DLTensorPythonFunction')

    @staticmethod
    def _function_wrapper_dlpack(batch_processing, function, num_outputs, *dlpack_inputs):
        if batch_processing:
            return PythonFunction.function_wrapper_batch(function,
                                                         num_outputs,
                                                         lambda x: x,
                                                         lambda x: x,
                                                         *dlpack_inputs)
        else:
            return PythonFunction.function_wrapper_per_sample(function,
                                                              num_outputs,
                                                              lambda x: x,
                                                              lambda x: x,
                                                              *dlpack_inputs)

    def __init__(self, function, num_outputs=1, device='cpu', synchronize_stream=True,
                 batch_processing=True, **kwargs):

        def func(*ts):
            return DLTensorPythonFunction._function_wrapper_dlpack(
                batch_processing, function, num_outputs, *ts)

        super(DLTensorPythonFunction,
              self).__init__(impl_name="DLTensorPythonFunctionImpl",
                             function=func,
                             num_outputs=num_outputs,
                             device=device,
                             synchronize_stream=synchronize_stream,
                             batch_processing=batch_processing,
                             **kwargs)
