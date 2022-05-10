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

import nvidia.dali.backend as _b
import nvidia.dali.internal as _internal
import nvidia.dali.ops as _ops
import nvidia.dali.tensors as _tensors
from nvidia.dali._utils.eager_util import _Classification, _transform_data_to_tensorlist


_statefull_operators = {
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
}


_generator_opeartors = {
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
    'random__CoinFlip',
    'random__Normal',
    'random__Uniform',
    'BatchPermutation',
}


_stateless_operators_cache = {}


def _eager_op_base_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperatorBase(op_class):
        def __init__(self, *, max_batch_size, device_id, **kwargs):
            super().__init__(**kwargs)

            self._spec.AddArg('device_id', device_id)
            self._spec.AddArg('max_batch_size', max_batch_size)

            for i in range(num_inputs):
                self._spec.AddInput(op_name+f'[{i}]', self._device)

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
            ValueError(f"Operator '{wrapper_name}' not registered for mixed.")

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

def _prep_inputs(inputs, batch_size):
    inputs = list(inputs)

    for i, input in enumerate(inputs):
        if not isinstance(input, (_tensors.TensorListCPU, _tensors.TensorListGPU)):
            inputs[i] = _transform_data_to_tensorlist(input, batch_size)
    
    return inputs

def _prep_kwargs(kwargs):
    for key, value in kwargs.items():
        kwargs[key] = _Classification(value, f'Argument {key}', to_constant=True).data

    return kwargs


def _wrap_stateless(op_class, op_name, wrapper_name):
    def wrapper(*inputs, **kwargs):
        _disqualify_arguments(wrapper_name, kwargs, _wrap_stateless.disqualified_arguments)

        # Preprocess kwargs to get batch_size.
        kwargs = _prep_kwargs(kwargs)
        init_args, call_args = _ops._separate_kwargs(kwargs, _tensors.TensorListCPU)
        batch_size = _choose_batch_size(inputs, kwargs.pop('batch_size', -1))

        # Preprocess inputs, try to convert each input to TensorList.
        inputs = _prep_inputs(inputs, batch_size)

        init_args['max_batch_size'] = batch_size
        init_args['device'], init_args['device_id'] = _choose_device(
            op_name, wrapper_name, inputs, kwargs.get('device'))

        key = op_name + str(sorted(init_args.items()))

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


def _wrap_eager_op(op_class, submodule, wrapper_name, wrapper_doc):
    op_name = op_class.schema_name
    op_schema = _b.TryGetSchema(op_name)
    if op_schema.IsDeprecated() or op_name in _statefull_operators or op_name in _generator_opeartors:
        # TODO(ksztenderski): For now only exposing stateless operators.
        return
    else:
        wrapper = _wrap_stateless(op_class, op_name, wrapper_name)

    # Exposing to eager.experimental module.
    eager_module = _internal.get_submodule(sys.modules[__name__], 'experimental')
    op_module = _internal.get_submodule(eager_module, submodule)

    if not hasattr(op_module, wrapper_name):
        wrapper.__name__ = wrapper_name
        wrapper.__qualname__ = wrapper_name
        wrapper.__doc__ = wrapper_doc

        if submodule:
            wrapper.__module__ = op_module.__name__

        setattr(op_module, wrapper_name, wrapper)
