# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
from queue import Queue

import nvidia.dali.fn as _fn
import nvidia.dali.pipeline as _pipeline
import nvidia.dali.tensors as _tensors
import nvidia.dali.types as _types
from nvidia.dali.data_node import DataNode as _DataNode, _arithm_op, _check
from nvidia.dali.external_source import _prep_data_for_feed_input
from nvidia.dali._utils.external_source_impl import \
    get_callback_from_source as _get_callback_from_source, \
    accepted_arg_count as _accepted_arg_count


class DataNodeDebug(_DataNode):
    """Wrapper class around Tensor, implementing all of the DataNode attributes."""

    def __init__(self, data, name, device, source):
        super().__init__(name, device, source)
        self._data = data

    def __str__(self):
        indent = ' ' * 4
        return f'DataNodeDebug(\n{indent}name="{self.name}",\n{indent}data={_tensors._tensorlist_to_string(self._data, indent + " " * 5)})'

    __repr__ = __str__

    def gpu(self):
        if self.device == 'gpu':
            return self
        return DataNodeDebug(self._data._as_gpu(), self.name, 'gpu', self.source)

    def get(self):
        return self._data

    def shape(self):
        return self._data.shape()

    def __add__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["add", self, other], {})

    def __radd__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["add", other, self], {})

    def __sub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["sub", self, other], {})

    def __rsub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["sub", other, self], {})

    def __mul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["mul", self, other], {})

    def __rmul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["mul", other, self], {})

    def __pow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["pow", self, other], {})

    def __rpow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["pow", other, self], {})

    def __truediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["fdiv", self, other], {})

    def __rtruediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["fdiv", other, self], {})

    def __floordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["div", self, other], {})

    def __rfloordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["div", other, self], {})

    def __neg__(self):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["minus", self], {})

    def __eq__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["eq", self, other], {})

    def __ne__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["neq", self, other], {})

    def __lt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["lt", self, other], {})

    def __le__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["leq", self, other], {})

    def __gt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["gt", self, other], {})

    def __ge__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["geq", self, other], {})

    def __and__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitand", self, other], {})

    def __rand__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitand", other, self], {})

    def __or__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitor", self, other], {})

    def __ror__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitor", other, self], {})

    def __xor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitxor", self, other], {})

    def __rxor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(_arithm_op, ["bitxor", other, self], {})


def _transform_data_to_tensorlist(data, batch_size, layout=None, device_id=None):
    data = _prep_data_for_feed_input(data, batch_size, layout, device_id)

    if isinstance(data, list):
        if isinstance(data[0], _tensors.TensorGPU):
            data = _tensors.TensorListGPU(data, layout or "")
        else:
            data = _tensors.TensorListCPU(data, layout or "")

    return data


class _ExternalSourceDebug:
    """Debug mode version of ExternalSource operator."""

    def __init__(
            self, source=None, num_outputs=None, batch_size=-1, cycle=None, name=None, layout=None,
            batch=None, batch_info=None):
        if name is not None and num_outputs is not None:
            raise ValueError("`num_outputs` is not compatible with named `ExternalSource`")

        callback, source_desc = _get_callback_from_source(source, cycle, batch_info or False)

        self._name = name
        self._layout = layout
        self._num_outputs = num_outputs
        self._batch = batch
        self._batch_size = batch_size
        self._callback = callback
        self._source_desc = source_desc
        self._batch_info = batch_info
        self._current_iter = 0
        self._current_sample = 0
        self._feed_inputs = Queue()

        if callback is not None:
            arg_count = _accepted_arg_count(callback)
            if arg_count not in [0, 1]:
                raise TypeError("External source callback must be a callable with 0 or 1 argument")
            self.accepts_arg = arg_count > 0

    def _callback_args(self, idx_in_batch, epoch_idx):
        if not self.accepts_arg:
            return ()
        if idx_in_batch is not None:
            arg = _types.SampleInfo(
                self._current_sample + idx_in_batch,
                idx_in_batch,
                self._current_iter,
                epoch_idx)
        elif self._batch_info:
            arg = _types.BatchInfo(
                self._current_iter,
                epoch_idx)
        else:
            arg = self._current_iter
        return (arg,)

    def _get_batch(self, epoch_idx):
        try:
            if self._batch:
                callback_out = self._callback(*self._callback_args(None, epoch_idx))
            else:
                callback_out = [self._callback(*self._callback_args(i, epoch_idx))
                                for i in range(self._batch_size)]
            self._current_sample += self._batch_size
            self._current_iter += 1
        except StopIteration:
            self._current_iter = 0
            self._current_sample = 0
            raise
        return callback_out

    def _feed_input(self, data, kwargs):
        if self._callback is not None:
            raise RuntimeError(f"Cannot use `feed_input` on the external source '{self._name}' with a `source`"
                               " argument specified.")

        self._feed_inputs.put((data, kwargs))

    def _fetch(self, epoch_idx):
        """Fetches data from callback or provided with feed_input."""

        def to_data_node_debug(data):
            data = _transform_data_to_tensorlist(data, self._batch_size, layout)
            device = 'gpu' if isinstance(data, _tensors.TensorListGPU) else 'cpu'

            return DataNodeDebug(data, self._name, device, self._source_desc)

        if self._callback is not None:
            callback_out = self._get_batch(epoch_idx)
            layout = self._layout
            if self._num_outputs is not None:
                raw_data = []
                for idx in range(self._num_outputs):
                    if self._batch:
                        raw_data.append(callback_out[idx])
                    else:
                        raw_data.append([callback_out[i][idx] for i in range(self._batch_size)])
            else:
                raw_data = callback_out
        else:
            raw_data, feed_input_params = self._feed_inputs.get()
            layout = feed_input_params.get('layout', self._layout)

        if self._num_outputs is not None:
            return [to_data_node_debug(data) for data in raw_data]

        return to_data_node_debug(raw_data)


class _PipelineDebug(_pipeline.Pipeline):
    """Debug mode for pipeline. Allows access to data inside the pipeline execution by wrapping all
     operators inside their pipelines"""

    def __init__(self, exec_func, **kwargs):
        super().__init__(**kwargs)
        kwargs['exec_pipelined'] = False
        kwargs['exec_async'] = False
        self._debug_on = False
        self._es_input_name = 'input_'
        self._es_kwarg_name = 'kwarg_'
        self._subpipeline_kwargs = kwargs
        self._subpipelines = {}
        self._external_sources = {}
        self._feed_input_data = {}
        self._subpipelines_built = False
        self._cur_subpipeline_id = -1
        self._exec_func = exec_func

        import numpy as np
        seed = kwargs.get('seed', -1)
        if seed < 0:
            seed  = np.random.randint(0, 2**32)
        self._seed_generator = np.random.default_rng(seed)

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

        self._debug_on = True
        self._cur_subpipeline_id = -1
        _pipeline.Pipeline.push_current(self)

        res = self._exec_func()
        if res is None:
            res = ()
        elif not isinstance(res, tuple):
            res = (res,)

        self._debug_on = False
        if not self._subpipelines_built:
            self._subpipelines_built = True
        _pipeline.Pipeline.pop_current()

        # Transforming all variables to TensorLists.
        return tuple([val.get() if isinstance(val, DataNodeDebug) else _tensors.TensorListCPU(
            np.tile(val, (self._max_batch_size, *[1]*np.array(val).ndim))) for val in res])

    def feed_input(self, data_node, data, **kwargs):
        """Pass data to an ExternalSource operator inside the pipeline.

        Refer to :meth:`Pipeline.feed_input() <nvidia.dali.Pipeline.feed_input>` for details."""
        if not self._built:
            raise RuntimeError("Pipeline must be built first.")
        if isinstance(data_node, str):
            name = data_node
        else:
            _check(data_node)
            name = data_node.name

        if name not in self._external_sources:
            # Saving data, because pipeline hasn't been run yet.
            if name not in self._feed_input_data:
                self._feed_input_data[name] = []
            self._feed_input_data[name].append((data, kwargs))
        else:
            self._external_sources[name]._feed_input(name, data, kwargs)

    @staticmethod
    def _classify_data(data):
        """ Returns tuple (use_external_source, device, unpacked data).
        Based on data type determines if we should use external_source and with which device. If the type can be
        recognized as a batch without being falsely categorized as such, it is. This includes lists of supported
        tensor-like objects e.g. numpy arrays (the only list not treated as a batch is a list of objects of
        primitive types), :class:`DataNodeDebug` and TensorLists.
        """

        def is_primitive_type(x):
            return isinstance(x, (int, float, bool, str))

        if isinstance(data, list):
            if any([is_primitive_type(d) for d in data]):
                return False, 'cpu', data

            device = None
            data_list = []
            for d in data:
                _, cur_device, val = _PipelineDebug._classify_data(d)
                if device is None:
                    device = cur_device
                data_list.append(val)
            return True, device, data_list
        else:
            if is_primitive_type(data) or _types._is_numpy_array(data) or \
                    _types._is_mxnet_array(data) or isinstance(data, _tensors.TensorCPU):
                return False, 'cpu', data
            if _types._is_torch_tensor(data):
                return False, 'gpu' if data.is_cuda else 'cpu', data
            if hasattr(data, '__cuda_array_interface__') or isinstance(data, _tensors.TensorGPU):
                return False, 'gpu', data
            if isinstance(data, DataNodeDebug):
                return True, data.device, data.get()
            if isinstance(data, _tensors.TensorListCPU):
                return True, 'cpu', data
            if isinstance(data, _tensors.TensorListGPU):
                return True, 'gpu', data

        return False, 'cpu', data

    def _create_subpipeline(self, op_wrapper, inputs, kwargs):
        """Creates pipeline wrapper around operator call (only in the first run).
        Each pipeline uses ExternalSource operators for processing inputs and kwargs.
        Returns pipeline and classification of data as batch.
        """

        inputs_external_source = [False]*len(inputs)
        kwargs_external_source = {}

        @_pipeline.pipeline_def(**self._subpipeline_kwargs)
        def pipe():
            inputs_preprocessed = []
            kwargs_preprocessed = {}
            for i, input in enumerate(inputs):
                to_external_source, device, _ = _PipelineDebug._classify_data(input)
                inputs_external_source[i] = to_external_source
                if to_external_source:
                    inputs_preprocessed.append(_fn.external_source(
                        name=f'{self._es_input_name}{i}', device=device))
                else:
                    inputs_preprocessed.append(input)

            # Parsing kwargs in similar way to inputs
            for key, value in kwargs.items():
                to_external_source, device, _ = _PipelineDebug._classify_data(value)
                kwargs_external_source[key] = to_external_source
                if to_external_source:
                    kwargs_preprocessed[key] = _fn.external_source(
                        name=f'{self._es_kwarg_name}{key}', device=device)
                else:
                    kwargs_preprocessed[key] = value

            if 'seed' not in kwargs_preprocessed and op_wrapper.__name__ != '_arithm_op':
                kwargs_preprocessed['seed'] = self._seed_generator.integers(0, 2**32)
            res = op_wrapper(*inputs_preprocessed, **kwargs_preprocessed)

            return tuple(res) if isinstance(res, list) else res

        p = pipe()
        p.build()
        return (p, inputs_external_source, kwargs_external_source)

    def _external_source(self, name=None, **kwargs):
        self._cur_subpipeline_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_subpipeline_id,)
        if not self._subpipelines_built:
            es = _ExternalSourceDebug(batch_size=self._max_batch_size, name=name, **kwargs)

            # feed_input all data collected after build and before run
            for (data, fi_kwargs) in self._feed_input_data.pop(name, []):
                es._feed_input(data, fi_kwargs)

            self._external_sources[key] = es

        if key in self._external_sources:
            return self._external_sources[key]._fetch(self._epoch_idx)
        else:
            raise RuntimeError(f"Unexpected operator 'ExternalSource'. Debug mode does not support"
                               " changing the order of operators executed within the pipeline.")

    def _run_subpipeline(self, pipe_tuple, op_name, inputs, kwargs):
        """Run pipeline wrapper of a single operator."""
        self._debug_on = False
        pipe, inputs_es, kwargs_es = pipe_tuple

        if len(inputs_es) != len(inputs):
            raise RuntimeError(f"Trying to use operator '{op_name}' with different number of inputs than when"
                               f" it was built. {len(inputs_es)} != {len(inputs)}")
        if len(kwargs_es) != len(kwargs.items()):
            raise RuntimeError(f"Trying to use operator '{op_name}' with different number of keyward arguments"
                               " than when it was built.")

        def unexpected_argument_msg(to_external_source):
            return f"recognized as {'batch' if to_external_source else 'constant'} but when built value" \
                f" in its place was recognized as {'constant' if to_external_source else 'batch'}"

        for i, input in enumerate(inputs):
            to_external_source, _, data = _PipelineDebug._classify_data(input)
            if to_external_source != inputs_es[i]:
                raise RuntimeError(
                    f"In operator '{op_name}' input {input} {unexpected_argument_msg(to_external_source)}.")
            if to_external_source:
                pipe.feed_input(f'{self._es_input_name}{i}', data)
        for key, value in kwargs.items():
            to_external_source, _, data = _PipelineDebug._classify_data(value)
            if to_external_source != kwargs_es[key]:
                raise RuntimeError(
                    f"In operator '{op_name}' argument '{key}' {unexpected_argument_msg(to_external_source)}.")
            if to_external_source:
                pipe.feed_input(f'{self._es_kwarg_name}{key}', data)

        res = pipe.run()
        res = tuple([DataNodeDebug(tensor, **data_node.__dict__)
                    for tensor, data_node in zip(res, pipe._graph_out)])

        self._debug_on = True

        if len(res) == 1:
            return res[0]

        return res

    def _wrap_op_call(self, op_wrapper, inputs, kwargs):
        self._cur_subpipeline_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_subpipeline_id,)
        if not self._subpipelines_built:
            self._subpipelines[key] = self._create_subpipeline(op_wrapper, inputs, kwargs)

        if key in self._subpipelines:
            return self._run_subpipeline(self._subpipelines[key], op_wrapper.__name__, inputs, kwargs)
        else:
            raise RuntimeError(f"Unexpected operator '{op_wrapper.__name__}'. Debug mode does not support"
                               " changing the order of operators executed within the pipeline.")
