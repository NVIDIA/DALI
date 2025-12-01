# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import backend as _b
from nvidia.dali import internal as _internal
from nvidia.dali import ops as _ops
from nvidia.dali import tensors as _tensors
from nvidia.dali import types as _types
from nvidia.dali.external_source import _prep_data_for_feed_input


# Classification of eager operators. Operators not assigned to any class are exposed as stateless.
# If you created a new operator and it is not stateless you should add it to the appropriate set.
# You should also add a coverage test for it in `dali/test/python/test_eager_coverage.py`.

# Stateful operators - rely on the internal state (return different outputs across iterations).
_stateful_operators = {
    "decoders__ImageRandomCrop",
    "experimental__decoders__ImageRandomCrop",
    "noise__Gaussian",
    "noise__SaltAndPepper",
    "noise__Shot",
    "segmentation__RandomMaskPixel",
    "segmentation__RandomObjectBBox",
    "Jitter",
    "ROIRandomCrop",
    "RandomBBoxCrop",
    "RandomResizedCrop",
    "RandomCropGenerator",
    "random__Beta",
    "random__Choice",
    "random__CoinFlip",
    "random__Normal",
    "random__Uniform",
    "BatchPermutation",
}


# Iterator operators - Python iterators of readers.
_iterator_operators = {
    "experimental__readers__Video",
    "experimental__readers__Fits",
    "readers__COCO",
    "readers__Caffe",
    "readers__Caffe2",
    "readers__File",
    "readers__MXNet",
    "readers__NemoAsr",
    "readers__Numpy",
    "readers__Sequence",
    "readers__TFRecord",
    "readers__Video",
    "readers__VideoResize",
    "readers__Webdataset",
}


# Operators not exposed in the eager mode.
_excluded_operators = {
    "readers__TFRecord",
    "TFRecordReader",
    "PythonFunction",
    "DLTensorPythonFunction",
    "TorchPythonFunction",
    "NumbaFunction",
    "experimental__inputs__Video",  # internal queue not supported in eager mode
}


def _transform_data_to_tensorlist(data, batch_size, layout=None, device_id=-1):
    data = _prep_data_for_feed_input(data, batch_size, layout, device_id)

    if isinstance(data, list):
        if isinstance(data[0], _tensors.TensorGPU):
            data = _tensors.TensorListGPU(data, layout or "")
        else:
            data = _tensors.TensorListCPU(data, layout or "")

    return data


class _Classification:
    """Classification of data's device and whether it is a batch.

    Based on data type determines if data should be treated as a batch and with which device.
    If the type can be recognized as a batch without being falsely categorized as such, it is.
    This includes lists of supported tensor-like objects e.g. numpy arrays (the only list not
    treated as a batch is a list of objects of primitive types), :class:`DataNodeDebug` and
    TensorLists.

    Args:
        data: Data to be classified.
        type_name (str): Representation of argument type (input or keyword).
        arg_constant_len (int): Only applicable for argument inputs that are of array type
            (e.g. numpy array). If -1 does not modify the data. For positive value works like
            `:func:types.Constant`, repeats the data `arg_constant_len` times.
    """

    def __init__(self, data, type_name, arg_constant_len=-1):
        from nvidia.dali._debug_mode import DataNodeDebug

        is_batch, device, extracted = self._classify_data(data, type_name, arg_constant_len)
        self.is_batch = is_batch
        self.device = device
        self.data = extracted
        self.was_data_node = isinstance(data, DataNodeDebug)
        self.original = data

    @staticmethod
    def _classify_data(data, type_name, arg_constant_len):
        """Returns tuple (is_batch, device, unpacked data)."""
        from nvidia.dali._debug_mode import DataNodeDebug

        def is_primitive_type(x):
            return isinstance(x, (int, float, bool, str))

        def classify_array_input(arr):
            if _types._is_numpy_array(arr):
                device = "cpu"
            elif _types._is_torch_tensor(arr):
                device = "gpu" if arr.is_cuda else "cpu"
            elif _types._is_mxnet_array(arr):
                device = "gpu" if "gpu" in str(arr.context) else "cpu"
            else:
                raise RuntimeError(f"Unsupported array type '{type(arr)}'.")

            return False, device, arr

        def classify_array_kwarg(arr):
            if _types._is_torch_tensor(arr):
                if arr.is_cuda:
                    arr = arr.cpu().numpy()
            elif _types._is_mxnet_array(arr):
                import mxnet as mx

                if "gpu" in str(arr.context):
                    arr = arr.copyto(mx.cpu())
            elif not _types._is_numpy_array(arr):
                raise RuntimeError(f"Unsupported array type '{type(arr)}'.")

            arr = _types._preprocess_constant_array_type(arr)
            arr = _tensors.TensorListCPU([_tensors.TensorCPU(arr)] * arg_constant_len)
            return True, "cpu", arr

        if isinstance(data, list):
            if len(data) == 0 or any([is_primitive_type(d) for d in data]):
                return False, "cpu", data

            is_batch_list = []
            device_list = []
            data_list = []

            for d in data:
                is_batch, device, val = _Classification._classify_data(d, type_name, -1)
                is_batch_list.append(is_batch)
                device_list.append(device)
                data_list.append(val)

            if any([device != device_list[0] for device in device_list]):
                raise RuntimeError(
                    f"{type_name} has batches of data on CPU and on GPU, " "which is not supported."
                )

            if all(is_batch_list):
                # Input set.
                return is_batch_list, device_list[0], data_list
            if not any(is_batch_list):
                # Converting to TensorList.
                return (
                    True,
                    device_list[0],
                    _transform_data_to_tensorlist(data_list, len(data_list)),
                )
            else:
                raise RuntimeError(f"{type_name} has inconsistent batch classification.")

        else:
            if isinstance(data, DataNodeDebug):
                return True, data.device, data.get()
            if isinstance(data, _tensors.TensorListCPU):
                return True, "cpu", data
            if isinstance(data, _tensors.TensorListGPU):
                return True, "gpu", data
            if is_primitive_type(data) or isinstance(data, _tensors.TensorCPU):
                return False, "cpu", data
            if _types._is_compatible_array_type(data):
                if arg_constant_len > 0:
                    # For call argument input repeats data `arg_constant_len` times to match
                    # the ConstantOp behavior.
                    return classify_array_kwarg(data)
                else:
                    return classify_array_input(data)
            if hasattr(data, "__cuda_array_interface__") or isinstance(data, _tensors.TensorGPU):
                return False, "gpu", data

        return False, "cpu", data


def _slice_tensorlist(data, size):
    """Constructs TensorList consisting of ``size`` first elements of ``data``."""

    return type(data)(list(data)[:size], layout=data.layout())


# TODO(klecki): Expose `nvidia.dali.math` in eager mode


def _arithm_op(name, *inputs):
    """Arithmetic operator function wrapper around ``eager.arithmetic_generic_op``. It is used
    for implementation of eager operators that are injected to TensorLists and for eager math
    operators.
    """
    batch_size = _choose_batch_size(inputs)
    inputs = [
        _Classification(input, f"Input {i}", arg_constant_len=batch_size).data
        for i, input in enumerate(inputs)
    ]
    categories_idxs, inputs, integers, reals = _ops._group_inputs(
        inputs, edge_type=(_tensors.TensorListCPU, _tensors.TensorListGPU)
    )
    input_desc = _ops._generate_input_desc(categories_idxs, integers, reals)

    if any(isinstance(input, _tensors.TensorListGPU) for input in inputs):
        device = "gpu"
    else:
        device = "cpu"

    if device == "gpu":
        inputs = list(
            input._as_gpu() if isinstance(input, _tensors.TensorListCPU) else input
            for input in inputs
        )

    init_args = {
        "device": device,
        "expression_desc": f"{name}({input_desc})",
        "integer_constants": integers,
        "real_constants": reals,
    }

    from nvidia.dali.experimental.eager import arithmetic_generic_op

    return arithmetic_generic_op(*inputs, **init_args)


# Implementations of TensorList operators, they are used directly in the backend.
def _add(self, other):
    return _arithm_op("add", self, other)


def _radd(self, other):
    return _arithm_op("add", other, self)


def _sub(self, other):
    return _arithm_op("sub", self, other)


def _rsub(self, other):
    return _arithm_op("sub", other, self)


def _mul(self, other):
    return _arithm_op("mul", self, other)


def _rmul(self, other):
    return _arithm_op("mul", other, self)


def _pow(self, other):
    return _arithm_op("pow", self, other)


def _rpow(self, other):
    return _arithm_op("pow", other, self)


def _truediv(self, other):
    return _arithm_op("fdiv", self, other)


def _rtruediv(self, other):
    return _arithm_op("fdiv", other, self)


def _floordiv(self, other):
    return _arithm_op("div", self, other)


def _rfloordiv(self, other):
    return _arithm_op("div", other, self)


def _neg(self):
    return _arithm_op("minus", self)


def _eq(self, other):
    return _arithm_op("eq", self, other)


def _ne(self, other):
    return _arithm_op("neq", self, other)


def _lt(self, other):
    return _arithm_op("lt", self, other)


def _le(self, other):
    return _arithm_op("leq", self, other)


def _gt(self, other):
    return _arithm_op("gt", self, other)


def _ge(self, other):
    return _arithm_op("geq", self, other)


def _and(self, other):
    return _arithm_op("bitand", self, other)


def _rand(self, other):
    return _arithm_op("bitand", other, self)


def _or(self, other):
    return _arithm_op("bitor", self, other)


def _ror(self, other):
    return _arithm_op("bitor", other, self)


def _xor(self, other):
    return _arithm_op("bitxor", self, other)


def _rxor(self, other):
    return _arithm_op("bitxor", other, self)


_stateless_operators_cache = {}


def _create_backend_op(spec, device, num_inputs, num_outputs, call_args_names, op_name):
    inp_device = "cpu" if device == "mixed" else device
    out_device = "gpu" if device == "mixed" else device

    for i in range(num_inputs):
        spec.AddInput(op_name + f"[{i}]", inp_device)

    for i in range(num_outputs):
        spec.AddOutput(op_name + f"_out[{i}]", out_device)

    for arg_name in call_args_names:
        spec.AddArgumentInput(arg_name, "")

    if device == "cpu":
        backend_op = _b.EagerOperatorCPU(spec)
    elif device == "gpu":
        backend_op = _b.EagerOperatorGPU(spec)
    elif device == "mixed":
        backend_op = _b.EagerOperatorMixed(spec)
    else:
        raise ValueError(f"Incorrect device type '{device}' in eager operator '{op_name}'.")

    return backend_op


def _eager_op_object_factory(op_class, op_name):
    """Creates eager operator class to use with objective ops-like API. For completeness,
    currently not used.
    """

    class EagerOperator(op_class):
        def __init__(self, **kwargs):
            self._batch_size = getattr(kwargs, "batch_size", -1)

            # Workaround for batch size deduction in _prep_args as we don't have inputs yet.
            kwargs["batch_size"] = 0

            _, init_args, _ = _prep_args(
                [], kwargs, op_name, op_name, _callable_op_factory.disqualified_arguments
            )
            device_id = init_args.pop("device_id")
            init_args.pop("max_batch_size")

            super().__init__(**init_args)

            self._spec.AddArg("device_id", device_id)
            self.built = False

        def __call__(self, *inputs, **kwargs):
            inputs, init_args, call_args = _prep_args(
                inputs, kwargs, op_name, op_name, _callable_op_factory.disqualified_arguments
            )

            if not self.built:
                num_outputs = self.schema.CalculateOutputs(
                    self._spec
                ) + self.schema.CalculateAdditionalOutputs(self._spec)

                self._spec.AddArg("max_batch_size", init_args["max_batch_size"])
                self._backend_op = _create_backend_op(
                    self._spec, self._device, len(inputs), num_outputs, call_args.keys(), op_name
                )
                self.built = True

            output = self._backend_op(inputs, kwargs)

            if len(output) == 1:
                return output[0]

            return output

    return EagerOperator


def _expose_eager_op_as_object(op_class, submodule):
    """Exposes eager operators as objects. Can be used if we decide to change eager API from
    functional to objective.
    """

    op_name = op_class.schema_name
    module = _internal.get_submodule("nvidia.dali.experimental.eager", submodule)
    op = _eager_op_object_factory(op_class, op_name)
    setattr(module, op_name, op)


def _eager_op_base_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperatorBase(op_class):
        def __init__(self, *, max_batch_size, device_id, **kwargs):
            super().__init__(**kwargs)

            self._spec.AddArg("device_id", device_id)
            self._spec.AddArg("max_batch_size", max_batch_size)

            num_outputs = self.schema.CalculateOutputs(
                self._spec
            ) + self.schema.CalculateAdditionalOutputs(self._spec)

            self._backend_op = _create_backend_op(
                self._spec, self._device, num_inputs, num_outputs, call_args_names, op_name
            )

    return EagerOperatorBase


def _create_module_class():
    """Creates a class imitating a module. Used for `rng_state` so we can have nested methods.
    E.g. `rng_state.random.normal`.
    """

    class Module:
        @classmethod
        def _submodule(cls, name):
            """Returns submodule, creates new if it does not exist."""
            if name not in cls._submodules:
                # Register a new submodule class (object representing submodule will be created in
                # the rng_state's constructor).
                cls._submodules[name] = _create_state_submodule(name)

            return cls._submodules[name]

        _submodules = {}

    return Module


def _create_state_submodule(name):
    """Creates a class imitating a submodule. It can contain methods and nested submodules.
    Used for submodules of rng_state, e.g. `rng_state.random`, `rng_state.noise`.
    """

    class StateSubmodule(_create_module_class()):
        def __init__(self, operator_cache, seed_generator):
            self._operator_cache = operator_cache
            self._seed_generator = seed_generator

            for name, submodule_class in StateSubmodule._submodules.items():
                # Adds nested submodules.
                setattr(self, name, submodule_class(self._operator_cache, self._seed_generator))

        __name__ = name

    return StateSubmodule


def _callable_op_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperator(_eager_op_base_factory(op_class, op_name, num_inputs, call_args_names)):
        def __call__(self, inputs, kwargs):
            # Here all kwargs are supposed to be TensorLists.
            output = self._backend_op(inputs, kwargs)

            if len(output) == 1:
                return output[0]

            return output

    return EagerOperator


_callable_op_factory.disqualified_arguments = {"bytes_per_sample_hint", "preserve", "seed"}


def _iterator_op_factory(op_class, op_name, num_inputs, call_args_names):
    class EagerOperator(_eager_op_base_factory(op_class, op_name, num_inputs, call_args_names)):
        def __init__(self, call_args, *, max_batch_size, **kwargs):
            pad_last_batch = kwargs.get("pad_last_batch", False)
            kwargs["pad_last_batch"] = True

            super().__init__(max_batch_size=max_batch_size, **kwargs)

            self._call_args = call_args
            self._iter = 0

            epoch_size = self._backend_op.reader_meta()["epoch_size"]
            self._num_iters = (epoch_size + max_batch_size - 1) // max_batch_size

            # Size of the last batch in an epoch.
            if pad_last_batch or epoch_size % max_batch_size == 0:
                self._last_batch_size = max_batch_size
            else:
                self._last_batch_size = epoch_size % max_batch_size

            assert isinstance(self._last_batch_size, int)

        def __next__(self):
            """Iterates over dataset once per epoch (last batch may not be full)."""

            if self._iter == self._num_iters:
                self._iter = 0
                raise StopIteration
            else:
                self._iter += 1
                outputs = self._backend_op([], self._call_args)

                if self._iter == self._num_iters:
                    # Return potentially partial batch at the end of an epoch.
                    outputs = [
                        _slice_tensorlist(tl_output, self._last_batch_size) for tl_output in outputs
                    ]

                if len(outputs) == 1:
                    outputs = outputs[0]

                return outputs

        def __iter__(self):
            return self

        def __len__(self):
            return self._num_iters

    return EagerOperator


_iterator_op_factory.disqualified_arguments = {
    "bytes_per_sample_hint",
    "preserve",
}


def _choose_device(op_name, wrapper_name, inputs, device_param):
    """Returns device type and device_id based on inputs and device_param."""

    input_device = ""

    if len(inputs) > 0:
        if any(isinstance(input, _tensors.TensorListGPU) for input in inputs):
            input_device = "gpu:0"
        else:
            input_device = "cpu"

    if device_param is None:
        # Select device type based on inputs.
        device_param = input_device if input_device else "cpu"

    sep_pos = device_param.find(":")

    # Separate device and device_id.
    if sep_pos != -1:
        device = device_param[:sep_pos]
        device_id = int(device_param[sep_pos + 1 :])
    else:
        device = device_param
        device_id = 0

    if device not in ["cpu", "gpu", "mixed"]:
        raise ValueError(f"Incorrect device type '{device}'.")

    return device, device_id


def _disqualify_arguments(op_name, kwargs, disqualified_args):
    for key in disqualified_args:
        if key in kwargs:
            raise RuntimeError(f"Argument '{key}' is not supported by eager operator '{op_name}'.")


def _choose_batch_size(inputs, batch_size=-1):
    """Returns batch size based on inputs and batch_size parameter."""

    if len(inputs) > 0:
        input_batch_size = -1

        for input in inputs:
            if hasattr(input, "__len__"):
                input_batch_size = len(input)
            if isinstance(input, (_tensors.TensorListCPU, _tensors.TensorListGPU)):
                # TensorList inputs have priority for choosing batch size.
                break

        if batch_size == -1:
            if input_batch_size == -1:
                raise RuntimeError("Could not deduce 'batch_size' from inputs.")
            batch_size = input_batch_size

        if input_batch_size != batch_size:
            raise ValueError(
                f"Requested batch_size={batch_size}, but input 0 has batch_size={input_batch_size}"
            )

    if batch_size == -1:
        raise RuntimeError(
            "Operators with no inputs need to have 'batch_size' parameter specified."
        )

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
                value, f"Argument {key}", arg_constant_len=batch_size
            ).data

        return kwargs

    _disqualify_arguments(wrapper_name, kwargs, disqualified_arguments)

    # Preprocess kwargs to get batch_size.
    batch_size = _choose_batch_size(inputs, kwargs.pop("batch_size", -1))
    kwargs = _prep_kwargs(kwargs, batch_size)
    init_args, call_args = _ops._separate_kwargs(kwargs, _tensors.TensorListCPU)

    # Preprocess inputs, try to convert each input to TensorList.
    inputs = _prep_inputs(inputs, batch_size)

    init_args["max_batch_size"] = batch_size
    init_args["device"], init_args["device_id"] = _choose_device(
        op_name, wrapper_name, inputs, kwargs.get("device")
    )

    return inputs, init_args, call_args


def _desc_call_args(inputs, args):
    """Returns string description of call arguments (inputs and input arguments) to use as part of
    the caching key."""
    return str([(inp.dtype, inp.layout(), len(inp[0].shape())) for inp in inputs]) + str(
        sorted(
            [
                (key, value.dtype, value.layout(), len(value[0].shape()))
                for key, value in args.items()
            ]
        )
    )


def _gen_cache_key(op_name, inputs, init_args, call_args):
    """Creating cache key consisting of operator name, description of inputs, input arguments
    and init args. Each call arg is described by dtype, layout and dim.
    """
    return op_name + _desc_call_args(inputs, call_args) + str(sorted(init_args.items()))


def _wrap_stateless(op_class, op_name, wrapper_name):
    """Wraps stateless Eager Operator in a function. Callable the same way as functions in fn API,
    but directly with TensorLists.
    """

    def wrapper(*inputs, **kwargs):
        inputs, init_args, call_args = _prep_args(
            inputs, kwargs, op_name, wrapper_name, _callable_op_factory.disqualified_arguments
        )

        key = _gen_cache_key(op_name, inputs, init_args, call_args)

        if key not in _stateless_operators_cache:
            _stateless_operators_cache[key] = _callable_op_factory(
                op_class, wrapper_name, len(inputs), call_args.keys()
            )(**init_args)

        return _stateless_operators_cache[key](inputs, call_args)

    return wrapper


def _wrap_stateful(op_class, op_name, wrapper_name):
    """Wraps stateful Eager Operator as method of a class. Callable the same way as functions in
    fn API, but directly with TensorLists.
    """

    def wrapper(self, *inputs, **kwargs):
        inputs, init_args, call_args = _prep_args(
            inputs, kwargs, op_name, wrapper_name, _callable_op_factory.disqualified_arguments
        )

        key = _gen_cache_key(op_name, inputs, init_args, call_args)

        if key not in self._operator_cache:
            # Creating a new operator instance with deterministically generated seed, so if we
            # preserve the order of operator calls in different instances of rng_state, they
            # return the same results.
            seed = self._seed_generator.integers(_wrap_stateful.seed_upper_bound)
            self._operator_cache[key] = _callable_op_factory(
                op_class, wrapper_name, len(inputs), call_args.keys()
            )(**init_args, seed=seed)

        return self._operator_cache[key](inputs, call_args)

    return wrapper


_wrap_stateful.seed_upper_bound = (1 << 31) - 1


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
            inputs, kwargs, op_name, wrapper_name, _iterator_op_factory.disqualified_arguments
        )

        op = _iterator_op_factory(op_class, wrapper_name, len(inputs), call_args.keys())(
            call_args, **init_args
        )

        return op

    return wrapper


def _get_rng_state_target_module(submodules):
    """Returns target module of rng_state. If a module did not exist, creates it."""
    from nvidia.dali.experimental import eager

    last_module = eager.rng_state
    for cur_module_name in submodules:
        # If nonexistent registers rng_state's submodule.
        cur_module = last_module._submodule(cur_module_name)
        last_module = cur_module

    return last_module


def _get_eager_target_module(parent_module, submodules, make_hidden):
    """Returns target module inside ``parent_module`` if specified, otherwise inside eager."""
    if parent_module is None:
        # Exposing to nvidia.dali.experimental.eager module.
        parent_module = _internal.get_submodule("nvidia.dali", "experimental.eager")
    else:
        # Exposing to experimental.eager submodule of the specified parent module.
        parent_module = _internal.get_submodule(sys.modules[parent_module], "experimental.eager")

    if make_hidden:
        op_module = _internal.get_submodule(parent_module, submodules[:-1])
    else:
        op_module = _internal.get_submodule(parent_module, submodules)

    return op_module


def _wrap_eager_op(op_class, submodules, parent_module, wrapper_name, wrapper_doc, make_hidden):
    """Exposes eager operator to the appropriate module
    (similar to :func:`nvidia.dali.fn._wrap_op`).
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

    if op_schema.IsDeprecated() or op_name in _excluded_operators:
        return
    elif op_name in _stateful_operators:
        wrapper = _wrap_stateful(op_class, op_name, wrapper_name)
        op_module = _get_rng_state_target_module(submodules)
    else:
        if op_name in _iterator_operators:
            wrapper = _wrap_iterator(op_class, op_name, wrapper_name)
        else:
            # If operator is not stateful, generator, deprecated or excluded expose it as stateless.
            wrapper = _wrap_stateless(op_class, op_name, wrapper_name)

        op_module = _get_eager_target_module(parent_module, submodules, make_hidden)

    if not hasattr(op_module, wrapper_name):
        wrapper.__name__ = wrapper_name
        wrapper.__qualname__ = wrapper_name
        wrapper.__doc__ = wrapper_doc
        wrapper._schema_name = op_name

        if submodules:
            wrapper.__module__ = op_module.__name__

        setattr(op_module, wrapper_name, wrapper)
