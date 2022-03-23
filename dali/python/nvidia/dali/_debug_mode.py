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
from collections import namedtuple
from queue import Queue

import nvidia.dali.backend as _b
import nvidia.dali.ops as _ops
import nvidia.dali.pipeline as _pipeline
import nvidia.dali.tensors as _tensors
import nvidia.dali.types as _types
from nvidia.dali.data_node import DataNode as _DataNode, _check
from nvidia.dali.external_source import _prep_data_for_feed_input
from nvidia.dali._utils.external_source_impl import \
    get_callback_from_source as _get_callback_from_source, \
    accepted_arg_count as _accepted_arg_count


_Classification = namedtuple('Classification', ['is_batch', 'device'])


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
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["add", self, other], {})

    def __radd__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["add", other, self], {})

    def __sub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["sub", self, other], {})

    def __rsub__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["sub", other, self], {})

    def __mul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["mul", self, other], {})

    def __rmul__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["mul", other, self], {})

    def __pow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["pow", self, other], {})

    def __rpow__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["pow", other, self], {})

    def __truediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["fdiv", self, other], {})

    def __rtruediv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["fdiv", other, self], {})

    def __floordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["div", self, other], {})

    def __rfloordiv__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["div", other, self], {})

    def __neg__(self):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["minus", self], {})

    def __eq__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["eq", self, other], {})

    def __ne__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["neq", self, other], {})

    def __lt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["lt", self, other], {})

    def __le__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["leq", self, other], {})

    def __gt__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["gt", self, other], {})

    def __ge__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["geq", self, other], {})

    def __and__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitand", self, other], {})

    def __rand__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitand", other, self], {})

    def __or__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitor", self, other], {})

    def __ror__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitor", other, self], {})

    def __xor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitxor", self, other], {})

    def __rxor__(self, other):
        return _PipelineDebug.current()._wrap_op_call(DataNodeDebug._arithm_op_name, ["bitxor", other, self], {})

    _arithm_op_name = 'ArithmeticGenericOp'


def _transform_data_to_tensorlist(data, batch_size, layout=None, device_id=None):
    if isinstance(data, DataNodeDebug):
        return data.get()

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
    """Debug mode for pipeline. Allows access to data inside the pipeline execution."""

    def __init__(self, exec_func, **kwargs):
        super().__init__(**kwargs)
        self._debug_on = False
        self._external_sources = {}
        self._feed_input_data = {}
        self._exec_func = exec_func
        self._cur_logical_id = -1
        self._operators = {}
        self._operators_built = False
        self._pipe = _b.PipelineDebug(
            self._max_batch_size, self._num_threads, self._device_id, self._set_affinity)

        import numpy as np
        seed = kwargs.get('seed', -1)
        if seed < 0:
            seed = np.random.randint(0, 2**32)
        self._seed_generator = np.random.default_rng(seed)

    def __enter__(self):
        raise RuntimeError("Currently pipeline in debug mode works only with `pipeline_def` decorator."
                           "Using `with` statement is not supported.")

    def build(self):
        """Build the pipeline.

        Symbolic version of build from the standard pipeline. In debug mode operators are built during
        the first run of the pipeline.

        Refer to :meth:`Pipeline.build() <nvidia.dali.Pipeline.build>` for details."""
        self._built = True

    def run(self):
        """Run the pipeline and return the result."""
        import numpy as np
        if not self._built:
            raise RuntimeError('Pipeline must be built first.')

        self._debug_on = True
        self._cur_logical_id = -1
        _pipeline.Pipeline.push_current(self)

        res = self._exec_func()
        if res is None:
            res = ()
        elif not isinstance(res, tuple):
            res = (res,)

        self._debug_on = False
        if not self._operators_built:
            self._operators_built = True
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
        """ Returns tuple (is_batch, device, unpacked data).

        Based on data type determines if data should be treated as a batch and with which device.
        If the type can be recognized as a batch without being falsely categorized as such, it is.
        This includes lists of supported tensor-like objects e.g. numpy arrays (the only list not
        treated as a batch is a list of objects of primitive types), :class:`DataNodeDebug` and
        TensorLists.
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
                    isinstance(data, _tensors.TensorCPU):
                return False, 'cpu', data
            if _types._is_torch_tensor(data):
                return False, 'gpu' if data.is_cuda else 'cpu', data
            if _types._is_mxnet_array(data):
                return False, 'gpu' if 'gpu' in str(data.context) else 'cpu', data
            if hasattr(data, '__cuda_array_interface__') or isinstance(data, _tensors.TensorGPU):
                return False, 'gpu', data
            if isinstance(data, DataNodeDebug):
                return True, data.device, data.get()
            if isinstance(data, _tensors.TensorListCPU):
                return True, 'cpu', data
            if isinstance(data, _tensors.TensorListGPU):
                return True, 'gpu', data

        return False, 'cpu', data

    @staticmethod
    def _separate_kwargs(kwargs):
        init_args, call_args, classification = {}, {}, {}

        for key, value in kwargs.items():
            is_batch, device, data = _PipelineDebug._classify_data(value)
            if is_batch:
                call_args[key] = data
            else:
                init_args[key] = data
            classification[key] = _Classification(is_batch, device)

        return init_args, call_args, classification

    @staticmethod
    def _create_arithm_op(init_args, name, *inputs):
        """Initializes arithmetic operator and returns inputs that are DataNodes."""

        categories_idxs, data_nodes, integers, reals = _ops._group_inputs(inputs)
        input_desc = _ops._generate_input_desc(categories_idxs, integers, reals)

        init_args['device'] = _ops._choose_device(data_nodes)
        init_args['expression_desc'] = "{}({})".format(name, input_desc)
        init_args['integer_constants'] = integers
        init_args['real_constants'] = reals

        return data_nodes

    def _create_op(self, op_name, inputs, kwargs):
        """Creates direct operator."""

        init_args, call_args, kwargs_classification = _PipelineDebug._separate_kwargs(kwargs)

        if op_name == 'ArithmeticGenericOp':
            inputs = _PipelineDebug._create_arithm_op(init_args, *inputs)
        else:
            if 'device' not in init_args and len(inputs) > 0:
                init_args['device'] = _PipelineDebug._classify_data(inputs[0])[1]

            if 'seed' not in init_args:
                init_args['seed'] = self._seed_generator.integers(0, 2**32)

        # Save inputs classification for later verification.
        inputs_classification = [_Classification(
            *_PipelineDebug._classify_data(input)[:2]) for input in inputs]

        op_helper = _ops._eager_op_factory(op_name)(**init_args)

        for arg_name in call_args.keys():
            # To use argument inputs OpSpec needs it specified (can be an empty placeholder).
            op_helper._spec.AddArgumentInput(arg_name, '')

        self._pipe.AddOperator(op_helper._spec, self._cur_logical_id)

        return op_helper, init_args, inputs_classification, kwargs_classification, len(inputs)

    def _external_source(self, name=None, **kwargs):
        self._cur_logical_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_logical_id,)
        if not self._operators_built:
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

    def _run_op_on_device(self, op_name, device, inputs, kwargs):
        if device == 'gpu':
            return self._pipe.RunOperatorGPU(self._cur_logical_id, inputs, kwargs)
        if device == 'cpu':
            return self._pipe.RunOperatorCPU(self._cur_logical_id, inputs, kwargs)
        if device == 'mixed':
            return self._pipe.RunOperatorMixed(self._cur_logical_id, inputs, kwargs)

        raise ValueError(f"Unknown device: '{device}' in operator '{op_name}'.")

    def _pack_to_data_node_debug(self, data, op_name):
        if isinstance(data, (list, tuple)):
            return [self._pack_to_data_node_debug(elem, op_name) for elem in data]

        return DataNodeDebug(data, op_name, 'gpu' if isinstance(data, _tensors.TensorListGPU) else 'cpu', self)

    def _run_op(self, op_tuple, op_name, inputs, kwargs):
        """Run a single operator."""

        def check_arg_len(expected_len, actual_len, args_type):
            if expected_len != actual_len:
                raise RuntimeError(f"Trying to use operator '{op_name}' with different number of {args_type} than when"
                                   f" it was built. Expected: {expected_len} {args_type}, got {actual_len}.")

        def check_batch_classification(expected, actual, arg_type, value):
            def classification_to_str(is_batch):
                return 'batch' if is_batch else 'constant'

            if expected != actual:
                expected_str = classification_to_str(expected)
                actual_str = classification_to_str(actual)

                raise RuntimeError(f"{arg_type} {value} for operator '{op_name}' is a {actual_str} "
                                   f"but was a {expected_str} when created.")

        def check_device_classification(expected, actual, arg_type, value):
            if expected != actual:
                raise RuntimeError(f"{arg_type} {value} for operator '{op_name}' is on '{actual}' "
                                   f"but was on '{expected}' when created.")

        op_helper, init_args, inputs_classification, kwargs_classification, expected_inputs_size = op_tuple
        op_device = init_args.get('device', 'cpu')

        check_arg_len(expected_inputs_size, len(inputs), 'inputs')
        check_arg_len(len(kwargs_classification), len(kwargs), 'keyward arguments')

        call_args = {}
        inputs = list(inputs)

        # Check inputs classification as batches and extract data from DataNodeDebugs.
        for i, input in enumerate(inputs):
            is_batch, device, data = _PipelineDebug._classify_data(input)

            check_batch_classification(inputs_classification[i].is_batch, is_batch, 'Input', i)
            check_device_classification(inputs_classification[i].device, device, 'Input', i)

            if device != ('gpu' if op_device == 'gpu' else 'cpu'):
                raise RuntimeError(f"Cannot call {op_device.upper()} operator '{op_name}' with "
                                   f"{device.upper()} input {i}.")

            inputs[i] = data

        # Check kwargs classification as batches and setup call args.
        for key, value in kwargs.items():
            is_batch, device, data = _PipelineDebug._classify_data(value)

            check_batch_classification(
                kwargs_classification[key].is_batch, is_batch, 'Argument', key)
            check_device_classification(kwargs_classification[key].device, device, 'Argument', key)

            if not is_batch and data != init_args[key]:
                raise RuntimeError(f"Argument '{key}' for operator '{op_name}' unexpectedly changed"
                                   f" value from '{init_args[key]}' to '{data}'")
            if is_batch:
                call_args[key] = data

        input_sets = op_helper._prep_input_sets(inputs)
        res = [self._run_op_on_device(op_name, op_device, input, call_args) for input in input_sets]

        if len(res) == 1:
            res = self._pack_to_data_node_debug(res[0], op_name)
        else:
            res = op_helper._repack_output_sets(res)
            res = self._pack_to_data_node_debug(res, op_name)

        if len(res) == 1:
            return res[0]

        return res

    @staticmethod
    def _extract_data_node_inputs(_, *inputs):
        """Extracts DataNodeDebugs from inputs for arithmetic operator and transforms data to GPU if needed."""
        data_nodes = []
        to_gpu = any([input.device == 'gpu' for input in inputs if isinstance(input, DataNodeDebug)])

        for input in inputs:
            if isinstance(input, DataNodeDebug):
                if to_gpu and input.device != 'gpu':
                    data_nodes.append(input.gpu())
                else:
                    data_nodes.append(input)

        return data_nodes

    def _wrap_op_call(self, op_name, inputs, kwargs):
        self._cur_logical_id += 1
        key = inspect.getframeinfo(
            inspect.currentframe().f_back.f_back)[:3] + (self._cur_logical_id,)
        if not self._operators_built:
            self._operators[key] = self._create_op(op_name, inputs, kwargs)

        if key in self._operators:
            if op_name == 'ArithmeticGenericOp':
                inputs = _PipelineDebug._extract_data_node_inputs(*inputs)
            return self._run_op(self._operators[key], op_name, inputs, kwargs)
        else:
            raise RuntimeError(f"Unexpected operator '{op_name}'. Debug mode does not support"
                               " changing the order of operators executed within the pipeline.")
