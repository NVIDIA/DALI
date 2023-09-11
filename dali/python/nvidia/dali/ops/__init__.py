# Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=no-member
import sys
import threading
import tree
import warnings
from itertools import count

import nvidia.dali.python_function_plugin
from nvidia.dali import backend as _b
from nvidia.dali import fn as _functional
from nvidia.dali import internal as _internal
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.types import (_type_name_convert_to_string, _type_convert_value,  # noqa: F401
                               _default_converter, _vector_element_type,  # noqa: F401
                               ScalarConstant as _ScalarConstant, Constant as _Constant)
from nvidia.dali import _conditionals

from nvidia.dali.ops import (_registry, _names, _docs, _operator_utils)  # noqa: F401

# reexpose what was previously visible:
from nvidia.dali.ops._registry import (cpu_ops, mixed_ops, gpu_ops, register_cpu_op,  # noqa: F401
                                       register_gpu_op)  # noqa: F401
from nvidia.dali.ops._names import (_op_name, _process_op_name, _schema_name)
from nvidia.dali.ops._operator_utils import (_build_input_sets, _repack_output_sets, )


class _OpCounter(object):
    # pylint: disable=too-few-public-methods
    _lock = threading.Lock()
    _op_count = count(0)

    def __init__(self):
        with self._lock:
            self._id = next(self._op_count)

    @property
    def id(self):
        return self._id


def _instantiate_constant_node(constant: _ScalarConstant, device: str):
    """Generate a DataNode (creating a Constant operator) based on the provided ScalarConstant.
    """
    return _Constant(device=device, value=constant.value, dtype=constant.dtype,
                     shape=constant.shape)


# TODO(klecki): The curse of multiple input sets and optimization prohibits us from using this
# code-path both for inputs and argument inputs.
def _handle_constant(value, device, input_name, op_name):
    """Handle promotion of possible constant value passed as (argument) input to an operator-backed
    DataNode. Pass-through if the value is a DataNode.

    Parameters
    ----------
    value : DataNode, ScalarConstant or value convertible to a constant op
        The value to be processed.
    device : str
        Target placement of constant node.
    input_name : int or str
        Position or name of the input, for error reporting purposes.
    op_name : str
        Name of the invoked operator, for error reporting purposes.

    Returns
    -------
    DataNode
        Either the same node as input or newly created DataNode representing the constant.

    Raises
    ------
    TypeError
        Error in case a constant was passed that is not possible to be converted by DALI.
    """
    if isinstance(value, _DataNode):
        return value
    if isinstance(value, _ScalarConstant):
        return _instantiate_constant_node(value, device)
    try:
        value = _Constant(value, device=device)
    except Exception as e:
        raise TypeError(f"when calling operator {op_name}: "
                        f"expected inputs of type `DataNode` or convertible to "
                        f"constant nodes. Received input `{input_name}` of type "
                        f"'{type(value).__name__}'.") from e
    return value


def _separate_kwargs(kwargs, arg_input_type=_DataNode):
    """Separates arguments into scalar arguments and argument inputs (data nodes or tensor lists),
    that were historically specified in __init__ and __call__ of operator class.

    Returns a pair of dictionaries of kwargs - the first for arguments (__init__), the second for
    argument inputs (__call__).

    Args:
        kwargs: Keyword arguments.
        arg_input_type: operator's argument input type, DataNode for pipeline mode, TensorListCPU
            for eager mode.
    """

    def is_arg_input_type(x):
        return isinstance(x, arg_input_type)

    def is_arg_input_or_name(name, value):
        if name == "device":
            return False
        if name == "ndim":
            return False
        if name == "name" or is_arg_input_type(value):
            return True
        if isinstance(value, (str, list, tuple, nvidia.dali.types.ScalarConstant)):
            return False
        return not nvidia.dali.types._is_scalar_value(value)

    def to_scalar(scalar):
        return scalar.value if isinstance(scalar, nvidia.dali.types.ScalarConstant) else scalar

    init_args = {}
    call_args = {}
    for name, value in kwargs.items():
        if value is None:
            continue
        if is_arg_input_or_name(name, value):
            call_args[name] = value
        else:
            init_args[name] = to_scalar(value)

    return init_args, call_args


def _handle_deprecations(schema, kwargs, op_name):
    """Handle deprecation of the named arguments (scalar arguments and argument inputs) specified
    to the operator.

    Based on the schema information the argument can be automatically renamed or dropped
    with appropriate warnings being issued. Errors are raised if both the old and new name of
    renamed argument are used.

    Parameters
    ----------
    schema : OpSchema
        Schema for the operator containing the deprecation information.
    kwargs : Dict
        Dictionary containing the arguments.
    op_name : str
        Name of the invoked operator, for error reporting purposes.

    Returns
    -------
    Dict
        Dictionary with arguments rearranged
    """
    arg_names = list(kwargs.keys())
    for arg_name in arg_names:
        if not schema.IsDeprecatedArg(arg_name):
            continue
        meta = schema.DeprecatedArgMeta(arg_name)
        new_name = meta['renamed_to']
        removed = meta['removed']
        msg = meta['msg']
        if new_name:
            if new_name in kwargs:
                raise TypeError(f"Operator {op_name} got an unexpected '{arg_name}' deprecated"
                                f" argument when '{new_name}' was already provided")
            kwargs[new_name] = kwargs[arg_name]
            del kwargs[arg_name]
        elif removed:
            del kwargs[arg_name]

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return kwargs


def _add_spec_args(schema, spec, kwargs):
    for key, value in kwargs.items():
        if value is None:
            # None is not a valid value for any argument type, so treat it
            # as if the argument was not supplied at all
            continue

        dtype = schema.GetArgumentType(key)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                spec.AddArgEmptyList(key, _vector_element_type(dtype))
                continue
        converted_value = _type_convert_value(dtype, value)
        spec.AddArg(key, converted_value)


class _OperatorInstance(object):

    def __init__(self, inputs, op, **kwargs):
        self._counter = _OpCounter()
        self._outputs = []
        self._op = op
        self._default_call_args = op._call_args
        self._spec = op.spec.copy()
        self._relation_id = self._counter.id

        if _conditionals.conditionals_enabled():
            inputs, kwargs = _conditionals.apply_conditional_split_to_args(inputs, kwargs)

        self._inputs = inputs

        spec_args, kwargs = _separate_kwargs(kwargs)
        _add_spec_args(op._schema, self._spec, spec_args)

        call_args = {**self._default_call_args}
        for k, v in kwargs.items():
            if v is None:
                # if an argument was specified in __init__ and in __call__ it is None, ignore it
                continue
            if k in self._default_call_args:
                raise ValueError("The argument `{}` was already specified in __init__.".format(k))
            call_args[k] = v

        name = call_args.get("name", None)
        if name is not None:
            self._name = name
        else:
            self._name = '__' + type(op).__name__ + "_" + str(self._counter.id)
        # Add inputs
        if inputs:
            for inp in inputs:
                if not isinstance(inp, _DataNode):
                    raise TypeError(
                        f"Expected inputs of type `DataNode`. Received input of type '{inp}'.")
                self._spec.AddInput(inp.name, inp.device)
        # Argument inputs
        for k in sorted(call_args.keys()):
            if k not in ["name"]:
                arg_inp = call_args[k]
                if arg_inp is None:
                    continue
                # Argument input constants are always placed on CPU
                arg_inp = _handle_constant(arg_inp, "cpu", k, type(op).__name__)

                _check_arg_input(op._schema, type(self._op).__name__, k)

                self._spec.AddArgumentInput(k, arg_inp.name)
                self._inputs = list(self._inputs) + [arg_inp]

        if self._op.schema.IsDeprecated():
            # TODO(klecki): how to know if this is fn or ops?
            msg = "WARNING: `{}` is now deprecated".format(_op_name(type(self._op).__name__, "fn"))
            use_instead = _op_name(self._op.schema.DeprecatedInFavorOf(), "fn")
            if use_instead:
                msg += ". Use `" + use_instead + "` instead."
            explanation = self._op.schema.DeprecationMessage()
            if explanation:
                msg += "\n" + explanation
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

    def check_args(self):
        self._op.schema.CheckArgs(self._spec)

    def generate_outputs(self):
        pipeline = _Pipeline.current()
        if pipeline is None and self._op.preserve:
            _Pipeline._raise_pipeline_required("Operators with side-effects ")
        # Add outputs
        if self._op.device == "gpu" or self._op.device == "mixed":
            output_device = "gpu"
        else:
            output_device = "cpu"

        num_output = (self._op.schema.CalculateOutputs(self._spec)
                      + self._op.schema.CalculateAdditionalOutputs(self._spec))

        if num_output == 0 and self._op.preserve:
            t_name = type(self._op).__name__ + "_id_" + str(self.id) + "_sink"
            pipeline.add_sink(_DataNode(t_name, output_device, self))
            return

        for i in range(num_output):
            t_name = self._name
            if num_output > 1:
                t_name += "[{}]".format(i)
            t = _DataNode(t_name, output_device, self)
            self._spec.AddOutput(t.name, t.device)
            if self._op.preserve:
                pipeline.add_sink(t)
            self.append_output(t)

    @property
    def id(self):
        return self._counter.id

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def unwrapped_outputs(self):
        if len(self._outputs) == 1:
            return self._outputs[0]
        else:
            return self._outputs

    @property
    def spec(self):
        return self._spec

    @property
    def name(self):
        return self._name

    @property
    def relation_id(self):
        return self._relation_id

    @relation_id.setter
    def relation_id(self, value):
        self._relation_id = value

    def append_output(self, output):
        self._outputs.append(output)


class _DaliOperatorMeta(type):

    @property
    def __doc__(self):
        return _docs._docstring_generator(self)


def _check_arg_input(schema, op_name, name):
    if name == "name":
        return
    if not schema.IsTensorArgument(name):
        expected_type_name = _type_name_convert_to_string(schema.GetArgumentType(name), False)
        raise TypeError(
            f"The argument `{name}` for operator `{op_name}` should not be a `DataNode` but a "
            f"{expected_type_name}")


def python_op_factory(name, schema_name=None):

    class Operator(metaclass=_DaliOperatorMeta):

        def __init__(self, *, device="cpu", **kwargs):
            schema_name = _schema_name(type(self))
            self._spec = _b.OpSpec(schema_name)
            self._schema = _b.GetSchema(schema_name)

            # Get the device argument. We will need this to determine
            # the device that our outputs will be stored on
            self._device = device
            self._spec.AddArg("device", self._device)

            kwargs, self._call_args = _separate_kwargs(kwargs)

            for k in self._call_args.keys():
                _check_arg_input(self._schema, type(self).__name__, k)

            if "preserve" in kwargs.keys():
                self._preserve = kwargs["preserve"]
                # we don't want to set "preserve" arg twice
                del kwargs["preserve"]
            else:
                self._preserve = False
            self._spec.AddArg("preserve", self._preserve)
            self._preserve = self._preserve or self._schema.IsNoPrune()

            # Check for any deprecated arguments that should be replaced or removed
            # TODO(klecki): Why can't we also handle deprecation of Argument Inputs?
            kwargs = _handle_deprecations(self._schema, kwargs, type(self).__name__)

            # Store the specified arguments
            _add_spec_args(self._schema, self._spec, kwargs)

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
            self._check_schema_num_inputs(inputs)

            inputs = _preprocess_inputs(inputs, self.__class__.__name__, self._device, self._schema)

            input_sets = _build_input_sets(inputs, self.__class__.__name__)

            # Create OperatorInstance for every input set
            op_instances = []
            for input_set in input_sets:
                op_instances.append(_OperatorInstance(input_set, self, **kwargs))
                op_instances[-1].generate_outputs()

            # Tie the instances together
            relation_id = op_instances[0].id
            for op in op_instances:
                op.relation_id = relation_id

            # If we don't have multiple input sets, flatten the result
            if len(op_instances) == 1:
                result = op_instances[0].unwrapped_outputs
            else:
                outputs = []
                for op in op_instances:
                    outputs.append(op.outputs)
                result = _repack_output_sets(outputs)
            if _conditionals.conditionals_enabled():
                if len(op_instances) != 1:
                    raise ValueError("Multiple input sets are not supported with conditional"
                                     " execution (when `enable_conditionals=True`)")
                _conditionals.register_data_nodes(result, input_sets[0], kwargs)
            return result

        def _check_schema_num_inputs(self, inputs):
            if len(inputs) < self._schema.MinNumInput() or len(inputs) > self._schema.MaxNumInput():
                raise ValueError(
                    f"Operator {type(self).__name__} expects "
                    f"from {self._schema.MinNumInput()} to {self._schema.MaxNumInput()} inputs, "
                    f"but received {len(inputs)}.")

    Operator.__name__ = str(name)
    Operator.schema_name = schema_name or Operator.__name__
    Operator.__call__.__doc__ = _docs._docstring_generator_call(Operator.schema_name)
    return Operator


def _wrap_op(op_class, submodule=[], parent_module=None):
    return _functional._wrap_op(op_class, submodule, parent_module,
                                _docs._docstring_generator_fn(op_class))


def _load_ops():
    _registry._discover_ops()
    _all_ops = _registry._all_registered_ops()
    ops_module = sys.modules[__name__]

    for op_reg_name in _all_ops:
        # TODO(klecki): Make this a function: _add_op(op_reg_name) and invoke it immediately
        # with register_xxx_op(). Now it relies on those class being present in this module
        # at the time of registration.
        schema = _b.TryGetSchema(op_reg_name)
        make_hidden = schema.IsDocHidden() if schema else False
        _, submodule, op_name = _process_op_name(op_reg_name, make_hidden)
        module = _internal.get_submodule(ops_module, submodule)
        if not hasattr(module, op_name):
            op_class = python_op_factory(op_name, op_reg_name)
            op_class.__module__ = module.__name__
            setattr(module, op_name, op_class)

            if op_name not in ["ExternalSource"]:
                _wrap_op(op_class, submodule)

            # The operator was inserted into nvidia.dali.ops.hidden module, let's import it here
            # so it would be usable, but not documented as coming from other module
            if make_hidden:
                parent_module = _internal.get_submodule(ops_module, submodule[:-1])
                setattr(parent_module, op_name, op_class)


def Reload():
    _load_ops()


def _load_readers_tfrecord():
    """After backend ops are loaded, load the TFRecord readers (if they are available).
    """
    from nvidia.dali.ops._operators import tfrecord

    if not tfrecord.tfrecord_enabled():
        return

    tfrecord._TFRecordReaderImpl.__call__.__doc__ = _docs._docstring_generator_call(
        "readers__TFRecord")

    _registry.register_cpu_op('readers__TFRecord')
    _registry.register_cpu_op('TFRecordReader')

    ops_module = sys.modules[__name__]

    for op_reg_name, op_class in [('readers__TFRecord', tfrecord.TFRecord),
                                  ('TFRecordReader', tfrecord.TFRecordReader)]:
        op_class.schema_name = op_reg_name
        _, submodule, op_name = _process_op_name(op_reg_name)
        module = _internal.get_submodule(ops_module, submodule)
        if not hasattr(module, op_name):
            op_class.__module__ = module.__name__
            setattr(module, op_name, op_class)
            _wrap_op(op_class, submodule)


def _choose_device(inputs):
    for input in inputs:
        if isinstance(input, (tuple, list)):
            if any(getattr(inp, "device", None) == "gpu" for inp in input):
                return "gpu"
        elif getattr(input, "device", None) == "gpu":
            return "gpu"
    return "cpu"


def _preprocess_inputs(inputs, op_name, device, schema=None):
    """Promote all scalar values in the inputs tuple into operator-backed DataNodes.

    This operation needs to be performed first, so we can have less duplicated constant nodes
    when dealing with multiple input sets.

    Parameters
    ----------
    inputs : tuple
        The inputs can contain one level nesting of Multiple Input Sets.
    """
    if isinstance(inputs, tuple):
        inputs = list(inputs)

    def is_input(x):
        if isinstance(x, (_DataNode, nvidia.dali.types.ScalarConstant)):
            return True
        # One level of nesting for Multiple Input Sets. It must be a List[DataNode/ScalarConstant]
        # with at least one DataNode.
        return (isinstance(x, (list))
                and any(isinstance(y, _DataNode) for y in x)
                and all(isinstance(y, (_DataNode, nvidia.dali.types.ScalarConstant)) for y in x))

    def get_input_device(schema, input_idx):
        default_input_device = "gpu" if device == "gpu" else "cpu"
        if schema:
            input_device = schema.GetInputDevice(input_idx) or default_input_device
        else:
            input_device = default_input_device
        return input_device

    def _promote_scalar_constant(value, input_device):
        """When ScalarConstant is encountered, promote it to a DataNode, otherwise do
        a pass-through.
        """
        if isinstance(value, _ScalarConstant):
            return _instantiate_constant_node(value, input_device)
        return value

    for idx, inp in enumerate(inputs):
        if not is_input(inp):
            try:
                inp = _Constant(inp, device=get_input_device(schema, idx))
            except Exception as ex:
                raise TypeError(f"when calling operator {op_name}: "
                                f"expected inputs of type `DataNode`, list of `DataNode` "
                                f"or convertible to constant nodes. Received "
                                f"input `{idx}` of type '{type(inp).__name__}'.") from ex

        if not isinstance(inp, _DataNode):
            dev = get_input_device(schema, idx)
            # Process the single ScalarConstant or list possibly containing ScalarConstants
            # and promote each of them into a DataNode
            inp = tree.map_structure(lambda val: _promote_scalar_constant(val, dev), inp)

        inputs[idx] = inp
    return inputs


# This must go at the end - the purpose of these imports is to expose the operators in
# nvidia.dali.ops module

# Expose just the ExternalSource class, the fn.external_source is exposed by hand in
# appropriate module.
from nvidia.dali.external_source import ExternalSource  # noqa: E402

ExternalSource.__module__ = __name__

# Expose the PythonFunction family of classes and generate the fn bindings for them
from nvidia.dali.ops._operators.python_function import (  # noqa: E402, F401
    PythonFunctionBase,  # noqa: F401
    PythonFunction, DLTensorPythonFunction, _dlpack_to_array,  # noqa: F401
    _dlpack_from_array)  # noqa: F401

_wrap_op(PythonFunction)
_wrap_op(DLTensorPythonFunction)

# Compose is only exposed for ops API, no fn bindings are generated
from nvidia.dali.ops._operators.compose import Compose  # noqa: E402, F401

_registry.register_cpu_op('Compose')
_registry.register_gpu_op('Compose')


from nvidia.dali.ops._operators.math import (_arithm_op, _group_inputs,  # noqa: E402, F401
                                             _generate_input_desc)  # noqa: F401


# Discover and generate bindings for all regular operators.
_load_ops()

# Load the TFRecord after the backend ops are processed, to wrap it conditionally, if it exists.
_load_readers_tfrecord()
