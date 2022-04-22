import sys

import nvidia.dali.backend as _b
import nvidia.dali.internal as _internal
import nvidia.dali.ops as _ops
import nvidia.dali.tensors as _tensors


_random_operators = {
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
    # Random and generators:
    'random__CoinFlip',
    'random__Normal',
    'random__Uniform',
    'BatchPermutation',
}
# gaussian_blur?


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
}

_stateless_operators_cache = {}


def _eager_op_factory(op_class):
    class EagerOperator(op_class):
        def __init__(self, *, max_batch_size, device_id, **kwargs):
            super().__init__(**kwargs)

            self._spec.AddArg('device_id', device_id)
            self._spec.AddArg('max_batch_size', max_batch_size)

            if self._device == 'cpu':
                self._backend_op = _b.EagerOperatorCPU(self._spec)
            elif self._device == 'gpu':
                self._backend_op = _b.EagerOperatorGPU(self._spec)
            elif self._device == 'mixed':
                self._backend_op = _b.EagerOperatorMixed(self._spec)
            else:
                raise ValueError(
                    f"Incorrect device type '{self._device}' in eager operator '{op_class.schema_name}'.")

        def __call__(self, inputs, kwargs):
            # Here all kwargs are supposed to be TensorLists.
            output = self._backend_op(inputs, kwargs)

            if len(output) == 1:
                return output[0]

            return output

    return EagerOperator


def _choose_device(op_name, inputs, device_param):
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
            ValueError(f"Operator '{op_name}' not registered for mixed.")

    return device, device_id


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


def _wrap_eager_op(op_class, submodule, wrapper_name, wrapper_doc):
    def wrapper(*inputs, **kwargs):
        init_args, call_args = _ops._separate_kwargs(kwargs, _tensors.TensorListCPU)

        init_args['max_batch_size'] = _choose_batch_size(inputs, kwargs.pop('batch_size', -1))
        init_args['device'], init_args['device_id'] = _choose_device(
            schema_name, inputs, kwargs.get('device'))

        key = str(sorted(init_args.items()))

        if key not in _stateless_operators_cache:
            _stateless_operators_cache[key] = _eager_op_factory(op_class)(**init_args)

        return _stateless_operators_cache[key](inputs, call_args)

    schema_name = op_class.schema_name
    op_schema = _b.TryGetSchema(schema_name)
    if op_schema.IsDeprecated() or schema_name in _random_operators or schema_name in _generator_opeartors:
        # TODO(ksztenderski): For now only exposing stateless operators.
        return

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
