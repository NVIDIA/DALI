# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import math

from nvidia.dali import backend as _b
from nvidia.dali import internal as _internal
from nvidia.dali import ops as _ops
from nvidia.dali import tensors as _tensors
from nvidia.dali._utils.eager_utils import _Classification, _transform_data_to_tensorlist, \
    _slice_tensorlist


# Classification of eager operators. Operators not assigned to any class are exposed as stateless.
# If you created a new operator and it is not stateless you should add it to the appropriate set.
# You should also add a coverage test for it in `nvidia.dali.test.python.test_eager_cpu_only`.

# Stateful operators - rely on the internal state (return different outputs across iterations).
_stateful_operators = {
    'decoders__ImageRandomCrop',
    'noise__Gaussian',
    'noise__SaltAndPepper',
    'noise__Shot',
    'segmentation__RandomMaskPixel',
    'segmentation__RandomObjectBBox',
    'FastResizeCropMirror',
    'Jitter',
    'ROIRandomCrop',
    'RandomBBoxCrop',
    'RandomResizedCrop',
    'ResizeCropMirror',
    'random__CoinFlip',
    'random__Normal',
    'random__Uniform',
    'BatchPermutation',
}


# Iterator operators - Python iterators of readers.
_iterator_operators = {
    'experimental__readers__Video',
    'readers__COCO',
    'readers__Caffe',
    'readers__Caffe2',
    'readers__File',
    'readers__MXNet',
    'readers__NemoAsr',
    'readers__Numpy',
    'readers__Sequence',
    'readers__TFRecord',
    'readers__Video',
    'readers__VideoResize',
    'readers__Webdataset',
}


# Operators not exposed in the eager mode.
_excluded_operators = {
    'readers__TFRecord',
    'TFRecordReader',
    'PythonFunction',
    'DLTensorPythonFunction',
    'TorchPythonFunction',
    'NumbaFunction',
}

_stateless_operators_cache = {}


def _eager_op_base_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperatorBase(op_class):
        def __init__(self, *, max_batch_size, device_id, **kwargs):
            super().__init__(**kwargs)

            self._spec.AddArg('device_id', device_id)
            self._spec.AddArg('max_batch_size', max_batch_size)

            for i in range(num_inputs):
                self._spec.AddInput(op_name + f'[{i}]', self._device)

            for arg_name in call_args_names:
                self._spec.AddArgumentInput(arg_name, '')

            if self._device == 'cpu':
                self._backend_op = _b.EagerOperatorCPU(self._spec)
            elif self._device == 'gpu':
                self._backend_op = _b.EagerOperatorGPU(self._spec)
            elif self._device == 'mixed':
                self._backend_op = _b.EagerOperatorMixed(self._spec)
            else:
                raise ValueError(
                    f"Incorrect device type '{self._device}' in eager operator '{op_name}'.")

    return EagerOperatorBase


def _stateless_op_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperator(_eager_op_base_factory(op_class, op_name, num_inputs, call_args_names)):
        def __call__(self, inputs, kwargs):
            # Here all kwargs are supposed to be TensorLists.
            output = self._backend_op(inputs, kwargs)

            if len(output) == 1:
                return output[0]

            return output

    return EagerOperator


def _iterator_op_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperator(_eager_op_base_factory(op_class, op_name, num_inputs, call_args_names)):
        def __init__(self, call_args, *, max_batch_size, **kwargs):
            pad_last_batch = kwargs.get('pad_last_batch', False)
            kwargs['pad_last_batch'] = True

            super().__init__(max_batch_size=max_batch_size, **kwargs)

            self._call_args = call_args
            self._iter = 0

            epoch_size = self._backend_op.reader_meta()['epoch_size']
            self._num_iters = math.ceil(epoch_size / max_batch_size)

            # Size of the last batch in an epoch.
            if pad_last_batch or epoch_size % max_batch_size == 0:
                self._last_batch_size = max_batch_size
            else:
                self._last_batch_size = epoch_size % max_batch_size

            assert isinstance(self._last_batch_size, int)

        def __next__(self):
            """ Iterates over dataset once per epoch (last batch may not be full). """

            if self._iter == self._num_iters:
                self._iter = 0
                raise StopIteration
            else:
                self._iter += 1
                outputs = self._backend_op([], self._call_args)

                if self._iter == self._num_iters:
                    # Return potentially partial batch at the end of an epoch.
                    outputs = [_slice_tensorlist(tl_output, self._last_batch_size)
                               for tl_output in outputs]

                if len(outputs) == 1:
                    outputs = outputs[0]

                return outputs

        def __iter__(self):
            return self

        def __len__(self):
            return self._num_iters

    return EagerOperator


def _choose_device(op_name, wrapper_name, inputs, device_param):
    """Returns device type and device_id based on inputs and device_param."""

    input_device = ''

    if len(inputs) > 0:
        if any(isinstance(input, _tensors.TensorListGPU) for input in inputs):
            input_device = 'gpu:0'
        else:
            input_device = 'cpu'

    if device_param is None:
        # Select device type based on inputs.
        device_param = input_device if input_device else 'cpu'

    sep_pos = device_param.find(':')

    # Separate device and device_id.
    if sep_pos != -1:
        device = device_param[:sep_pos]
        device_id = int(device_param[sep_pos + 1:])
    else:
        device = device_param
        device_id = 0

    if device == 'cpu' and input_device == 'gpu':
        raise ValueError("An operator with device='cpu' cannot accept GPU inputs.")

    if device != 'cpu' and device != 'gpu':
        raise ValueError(f"Incorrect device type '{device}'.")

    if input_device == 'cpu' and device == 'gpu':
        if op_name in _ops._mixed_ops:
            device = 'mixed'
        else:
            raise ValueError(f"Operator '{wrapper_name}' not registered for mixed.")

    return device, device_id


def _disqualify_arguments(op_name, kwargs, disqualified_args):
    for key in disqualified_args:
        if key in kwargs:
            raise RuntimeError(f"Argument '{key}' is not supported by eager operator '{op_name}'.")


def _choose_batch_size(inputs, batch_size):
    """Returns batch size based on inputs and batch_size parameter."""

    if len(inputs) > 0:
        input_batch_size = len(inputs[0])

        if batch_size == -1:
            batch_size = input_batch_size

        if input_batch_size != batch_size:
            raise ValueError(
                f"Requested batch_size={batch_size}, but input 0 has batch_size={input_batch_size}")

    if batch_size == -1:
        raise RuntimeError(
            "Operators with no inputs need to have 'batch_size' parameter specified.")

    return batch_size


def _prep_args(inputs, kwargs, op_name, wrapper_name, disqualified_arguments):
    def _prep_inputs(inputs, batch_size):
        inputs = list(inputs)

        for i, input in enumerate(inputs):
            if not isinstance(input, (_tensors.TensorListCPU, _tensors.TensorListGPU)):
                inputs[i] = _transform_data_to_tensorlist(input, batch_size)

        return inputs

    def _prep_kwargs(kwargs, batch_size):
        for key, value in kwargs.items():
            kwargs[key] = _Classification(
                value, f'Argument {key}', arg_constant_len=batch_size).data

        return kwargs

    _disqualify_arguments(wrapper_name, kwargs, disqualified_arguments)

    # Preprocess kwargs to get batch_size.
    batch_size = _choose_batch_size(inputs, kwargs.pop('batch_size', -1))
    kwargs = _prep_kwargs(kwargs, batch_size)
    init_args, call_args = _ops._separate_kwargs(kwargs, _tensors.TensorListCPU)

    # Preprocess inputs, try to convert each input to TensorList.
    inputs = _prep_inputs(inputs, batch_size)

    init_args['max_batch_size'] = batch_size
    init_args['device'], init_args['device_id'] = _choose_device(
        op_name, wrapper_name, inputs, kwargs.get('device'))

    return inputs, init_args, call_args


def _desc_call_args(inputs, args):
    """Returns string description of call arguments (inputs and input arguments) to use as part of
    the caching key."""
    return str([(inp.dtype, inp.layout(), len(inp[0].shape())) for inp in inputs]) + str(sorted(
        [(key, value.dtype, value.layout(), len(value[0].shape())) for key, value in args.items()]))


def _wrap_stateless(op_class, op_name, wrapper_name):
    """Wraps stateless Eager Operator in a function. Callable the same way as functions in fn API,
    but directly with TensorLists.
    """
    def wrapper(*inputs, **kwargs):
        inputs, init_args, call_args = _prep_args(
            inputs, kwargs, op_name, wrapper_name, _wrap_stateless.disqualified_arguments)

        # Creating cache key consisting of operator name, description of inputs, input arguments
        # and init args. Each call arg is described by dtype, layout and dim.
        key = op_name + _desc_call_args(inputs, call_args) + str(sorted(init_args.items()))

        if key not in _stateless_operators_cache:
            _stateless_operators_cache[key] = _stateless_op_factory(
                op_class, wrapper_name, len(inputs), call_args.keys())(**init_args)

        return _stateless_operators_cache[key](inputs, call_args)

    return wrapper


_wrap_stateless.disqualified_arguments = {
    'bytes_per_sample_hint',
    'preserve',
    'seed'
}


def _wrap_iterator(op_class, op_name, wrapper_name):
    """Wraps reader Eager Operator in a Python iterator.
    
    Example:
        >>> for file, label in eager.readers.file(file_root=file_path, batch_size=8):
        ...     # file and label are batches of size 8 (TensorLists).
        ...     print(file)
    """
    def wrapper(*inputs, **kwargs):
        if len(inputs) > 0:
            raise ValueError("Iterator type eager operators should not receive any inputs.")

        inputs, init_args, call_args = _prep_args(
            inputs, kwargs, op_name, wrapper_name, _wrap_iterator.disqualified_arguments)

        op = _iterator_op_factory(op_class, wrapper_name, len(inputs),
                                  call_args.keys())(call_args, **init_args)

        return op

    return wrapper


_wrap_iterator.disqualified_arguments = {
    'bytes_per_sample_hint',
    'preserve',
}


def _wrap_eager_op(op_class, submodule, parent_module, wrapper_name, wrapper_doc, make_hidden):
    """Exposes eager operator to the appropriate module (similar to :func:`nvidia.dali.fn._wrap_op`).
    Uses ``op_class`` for preprocessing inputs and keyword arguments and filling OpSpec for backend
    eager operators.

    Args:
        op_class: Op class to wrap.
        submodule: Additional submodule (scope).
        parent_module (str): If set to None, the wrapper is placed in nvidia.dali.experimental.eager
            module, otherwise in a specified parent module.
        wrapper_name: Wrapper name (the same as in fn API).
        wrapper_doc (str): Documentation of the wrapper function.
        make_hidden (bool): If operator is hidden, we should extract it from hidden submodule.
    """
    op_name = op_class.schema_name
    op_schema = _b.TryGetSchema(op_name)
    if op_schema.IsDeprecated() or op_name in _excluded_operators or op_name in _stateful_operators:
        # TODO(ksztenderski): For now only exposing stateless and iterator operators.
        return
    elif op_name in _iterator_operators:
        wrapper = _wrap_iterator(op_class, op_name, wrapper_name)
    else:
        # If operator is not stateful or a generator expose it as stateless.
        wrapper = _wrap_stateless(op_class, op_name, wrapper_name)

    if parent_module is None:
        # Exposing to experimental.eager module.
        parent_module = sys.modules[__name__]
    else:
        # Exposing to experimental.eager submodule of the specified parent module.
        parent_module = _internal.get_submodule(sys.modules[parent_module], 'experimental.eager')

    if make_hidden:
        op_module = _internal.get_submodule(parent_module, submodule[:-1])
    else:
        op_module = _internal.get_submodule(parent_module, submodule)

    if not hasattr(op_module, wrapper_name):
        wrapper.__name__ = wrapper_name
        wrapper.__qualname__ = wrapper_name
        wrapper.__doc__ = wrapper_doc
        wrapper._schema_name = op_schema

        if submodule:
            wrapper.__module__ = op_module.__name__

        setattr(op_module, wrapper_name, wrapper)
