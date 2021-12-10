# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.backend as _b
import nvidia.dali.fn as _fn
import nvidia.dali.pipeline as _pipeline
import nvidia.dali.tensors as _Tensors
import nvidia.dali.types as _types
from nvidia.dali.data_node import DataNode as _DataNode, _arithm_op
import inspect
from functools import partial

from torch.functional import Tensor


class DataNodeDebug(_DataNode):
    """Wrapper class around Tensor, implementing all of the DataNode attributes."""

    def __init__(self, data, name, device, source):
        super().__init__(name, device, source)
        self._data = data

    def gpu(self):
        if self.device == 'gpu':
            return self
        return DataNodeDebug(self._data._as_gpu(), self.name, 'gpu', self.source)

    def get(self):
        return self._data

    def __add__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, 'add', self, other)

    def __radd__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "add", other, self)

    def __sub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "sub", self, other)

    def __rsub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "sub", other, self)

    def __mul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "mul", self, other)

    def __rmul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "mul", other, self)

    def __pow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "pow", self, other)

    def __rpow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "pow", other, self)

    def __truediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "fdiv", self, other)

    def __rtruediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "fdiv", other, self)

    def __floordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "div", self, other)

    def __rfloordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "div", other, self)

    def __neg__(self):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "minus", self)

    def __eq__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "eq", self, other)

    def __ne__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "neq", self, other)

    def __lt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "lt", self, other)

    def __le__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "leq", self, other)

    def __gt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "gt", self, other)

    def __ge__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "geq", self, other)

    def __and__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitand", self, other)

    def __rand__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitand", other, self)

    def __or__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitor", self, other)

    def __ror__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitor", other, self)

    def __xor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitxor", self, other)

    def __rxor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, "bitxor", other, self)


class _PipelineDebugRunner():
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        self.pipeline._debug_on = True
        self.pipeline._external_source_debug = True
        self.pipeline._cur_subpipeline_id = -1
        return super(type(self.pipeline), self.pipeline).__enter__()

    def __exit__(self, *exc):
        self.pipeline._debug_on = False
        self.pipeline._external_source_debug = False
        if not self.pipeline._subpipelines_built:
            self.pipeline._subpipelines_built = True
        return super(type(self.pipeline), self.pipeline).__exit__(*exc)


class _PipelineDebug(_pipeline.Pipeline):
    """Debug mode for pipeline. Allows access to data inside the pipeline execution by wrapping all
     operators inside their pipelines"""

    def __init__(self, exec_func, **kwargs):
        kwargs['debug'] = False
        super().__init__(**kwargs)
        kwargs['exec_pipelined'] = False
        kwargs['exec_async'] = False
        self._debug_on = False
        self._external_source_debug = False
        self._es_input_name = 'input_'
        self._es_kwarg_name = 'kwarg_'
        self._subpipeline_kwargs = kwargs
        self._subpipelines = {}
        self._external_source_pipelines = {}
        self._feed_input_data = {}
        self._subpipelines_built = False
        self._cur_subpipeline_id = -1
        self._exec_func = exec_func
        self._runner = _PipelineDebugRunner(self)

    def __enter__(self):
        raise RuntimeError("Currently pipeline in debug mode works only with `pipeline_def` decorator."
                           "Using `with` statement is not supported.")

    def build(self):
        """Build the pipeline.
        
        Refer to :meth:`Pipeline.build() <nvidia.dali.Pipeline.build>` for details."""
        self._built = True

    def run(self):
        """Run the pipeline and return the result."""
        import numpy as np
        if not self._built:
            raise RuntimeError('Pipeline must be built first.')
        with self._runner:
            res = self._exec_func()
            if res is None:
                res = ()
            elif not isinstance(res, tuple):
                res = (res,)

            # Transforming all variables to TensorLists.
            return tuple([val.get() if isinstance(val, DataNodeDebug) else _Tensors.TensorListCPU(
                np.tile(val, (self._max_batch_size, *[1]*np.array(val).ndim))) for val in res])

    def feed_input(self, data_node, data, **kwargs):
        """Pass data to an ExternalSource operator inside the pipeline.
        
        Refer to :meth:`Pipeline.feed_input() <nvidia.dali.Pipeline.feed_input>` for details."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")

        if data_node not in self._external_source_pipelines:
            # Saving data, because pipeline hasn't been run yet.
            if data_node not in self._feed_input_data:
                self._feed_input_data[data_node] = []
            self._feed_input_data[data_node].append((data, kwargs))
        else:
            self._external_source_pipelines[data_node].feed_input(data_node, data, **kwargs)

    @staticmethod
    def _classify_data(data):
        # Returns tuple (use_external_source, device, unpacked data).

        def is_primitive_type(x):
            return isinstance(x, (int, float, bool, str))

        if isinstance(data, list):
            if any([is_primitive_type(d) for d in data]):
                return False, 'cpu', data

            device = None
            data_list = []
            for d in data:
                _, cur_device, val = _PipelineDebug._classify_data(d)
                if device == None:
                    device = cur_device
                data_list.append(val)
            return True, device, data_list
        else:
            if is_primitive_type(data) or _types._is_numpy_array(data) or _types._is_mxnet_array(data) or isinstance(data, _Tensors.TensorCPU):
                return False, 'cpu', data
            if _types._is_torch_tensor(data):
                return False, 'gpu' if data.is_cuda else 'cpu', data
            if hasattr(data, '__cuda_array_interface__') or isinstance(data, _Tensors.TensorGPU):
                return False, 'gpu', data
            if isinstance(data, DataNodeDebug):
                return True, data.device, data.get()
            if isinstance(data, _Tensors.TensorListCPU):
                return True, 'cpu', data
            if isinstance(data, _Tensors.TensorListGPU):
                return True, 'gpu', data

        return False, 'cpu', data

    def _create_subpipeline(self, op_wrapper, inputs, **kwargs):
        """Creates pipeline wrapper around operator call (only once)."""

        @_pipeline.pipeline_def(**self._subpipeline_kwargs)
        def pipe():
            inputs_preprocessed = []
            kwargs_preprocessed = {}
            for i, input in enumerate(inputs):
                to_external_source, device, _ = _PipelineDebug._classify_data(input)
                if to_external_source:
                    inputs_preprocessed.append(_fn.external_source(
                        name=f'{self._es_input_name}{i}', device=device))
                else:
                    inputs_preprocessed.append(input)

            # Parsing kwargs in similar way to inputs
            for key, value in kwargs.items():
                to_external_source, device, _ = _PipelineDebug._classify_data(value)
                if to_external_source:
                    kwargs_preprocessed[key] = _fn.external_source(
                        name=f'{self._es_kwarg_name}{key}', device=device)
                else:
                    kwargs_preprocessed[key] = value

            res = op_wrapper(*inputs_preprocessed, **kwargs_preprocessed)

            return tuple(res) if isinstance(res, list) else res

        self._external_source_debug = False
        p = pipe()
        p.build()
        self._external_source_debug = True
        return p

    def _external_source(self, op_wrapper, name, *inputs, **kwargs):
        # TODO(ksztenderski): Possibly remove this wrapper to avoid running data through the backend and back.
        @_pipeline.pipeline_def(**self._subpipeline_kwargs)
        def external_source_pipeline():
            res = op_wrapper(*inputs, name=name, **kwargs)
            return tuple(res) if isinstance(res, list) else res

        self._cur_subpipeline_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_subpipeline_id,)
        if not self._subpipelines_built:
            self._external_source_debug = False
            p = external_source_pipeline()
            p.build()

            # feed_input all data collected after build and before run
            for (data, fi_kwargs) in self._feed_input_data.pop(name, []):
                p.feed_input(name, data, **fi_kwargs)

            self._subpipelines[key] = p
            self._external_source_pipelines[name] = p
            self._external_source_debug = True

        if key in self._subpipelines:
            return self._run_subpipeline(self._subpipelines[key], (), {})
        else:
            raise RuntimeError(f'Unexpected operator {op_wrapper.__name__}. Debug mode does not support'
                               ' changing the order of operators executed within the pipeline.')

    def _run_subpipeline(self, pipe, inputs, kwargs):
        """Run pipeline wrapper of a single operator."""
        self._debug_on = False

        for i, input in enumerate(inputs):
            to_external_source, _, data = _PipelineDebug._classify_data(input)
            if to_external_source:
                pipe.feed_input(f'{self._es_input_name}{i}', data)
        for key, value in kwargs.items():
            to_external_source, _, data = _PipelineDebug._classify_data(value)
            if to_external_source:
                pipe.feed_input(f'{self._es_kwarg_name}{key}', data)

        res = pipe.run()
        res = tuple([DataNodeDebug(tensor, **data_node.__dict__)
                    for tensor, data_node in zip(res, pipe._graph_out)])

        self._debug_on = True

        if len(res) == 1:
            return res[0]

        return res

    def _wrap_op_call(self, op_wrapper, *inputs, **kwargs):
        self._cur_subpipeline_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_subpipeline_id,)
        if not self._subpipelines_built:
            self._subpipelines[key] = self._create_subpipeline(op_wrapper, inputs, **kwargs)

        if key in self._subpipelines:
            return self._run_subpipeline(self._subpipelines[key], inputs, kwargs)
        else:
            raise RuntimeError(f'Unexpected operator {op_wrapper.__name__}. Debug mode does not support'
                               ' changing the order of operators executed within the pipeline.')
