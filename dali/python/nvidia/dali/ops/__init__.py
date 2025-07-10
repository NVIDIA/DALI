# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import weakref
from itertools import count

import nvidia.dali.python_function_plugin
from nvidia.dali import backend as _b
from nvidia.dali import fn as _functional
from nvidia.dali import internal as _internal
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.pipeline import Pipeline as _Pipeline
from nvidia.dali.types import (  # noqa: F401
    _type_name_convert_to_string,
    _type_convert_value,
    _default_converter,
    _vector_element_type,
    ScalarConstant as _ScalarConstant,
    Constant as _Constant,
)
from nvidia.dali import _conditionals

from nvidia.dali.ops import _registry, _names, _docs, _operator_utils  # noqa: F401

from nvidia.dali._utils import dali_trace as _dali_trace

# reexpose what was previously visible:
from nvidia.dali.ops._registry import (  # noqa: F401
    cpu_ops,
    mixed_ops,
    gpu_ops,
    register_cpu_op,
    register_gpu_op,
)
from nvidia.dali.ops._names import _op_name, _process_op_name, _schema_name
from nvidia.dali.ops._operator_utils import (
    _build_input_sets,
    _repack_output_sets,
)


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
    """Generate a DataNode (creating a Constant operator) based on the provided ScalarConstant."""
    return _Constant(
        device=device, value=constant.value, dtype=constant.dtype, shape=constant.shape
    )


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
        raise TypeError(
            f"when calling operator `{op_name}`: "
            f"expected inputs of type 'DataNode'` or convertible to "
            f"constant nodes. Received input `{input_name}` of type "
            f"'{type(value).__name__}'."
        ) from e
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

    def is_arg_input(name, value):
        if name == "device":
            return False
        if name == "ndim":
            return False
        if is_arg_input_type(value):
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
        if is_arg_input(name, value):
            call_args[name] = value
        else:
            init_args[name] = to_scalar(value)

    return init_args, call_args


def _handle_arg_deprecations(schema, kwargs, op_name):
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
        meta = schema.DeprecatedArgInfo(arg_name)
        new_name = meta["renamed_to"]
        removed = meta["removed"]
        msg = meta["msg"]
        if new_name:
            if new_name in kwargs:
                raise TypeError(
                    f"Operator `{op_name}` got an unexpected '{arg_name}' deprecated"
                    f" argument when '{new_name}' was already provided"
                )
            kwargs[new_name] = kwargs[arg_name]
            del kwargs[arg_name]
        elif removed:
            del kwargs[arg_name]

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return kwargs


def _handle_op_deprecation(schema, module, display_name):
    if schema.IsDeprecated():
        msg = f"WARNING: `{module}.{display_name}` is now deprecated."
        replacement = schema.DeprecatedInFavorOf()
        if replacement:
            use_instead = _op_name(replacement, "fn")
            msg += f" Use `nvidia.dali.fn.{use_instead}` instead."
        explanation = schema.DeprecationMessage()
        if explanation:
            msg += "\n" + explanation
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.warn(msg, DeprecationWarning, stacklevel=2)


def _resolve_double_definitions(current, previous, keep_old=True):
    """Unify the arguments (or argument inputs) between __call__ and __init__ - ops API only.

    If the same argument was passed in both invocation it should result in error, with exception,
    of passing `None` in `__call__`.

    Parameters
    ----------
    current : Dict
        Parameters passed in __call__
    previous : Dict
        Parameters initially passed in __init__
    keep_old : bool
        If the old parameters should be returned. Otherwise they just are checked against
        double definition

    Returns
    -------
    Dict
        Merged dictionaries after error handling.
    """
    merged_result = {**previous}  # copy the dict
    new_result = {}
    for k, v in current.items():
        if v is None:
            # if an argument was specified in __init__ and in __call__ it is None, ignore it
            continue
        if k in previous:
            raise ValueError(f"The argument `{k}` was already specified in __init__.")
        merged_result[k] = v
        new_result[k] = v

    if keep_old:
        return merged_result
    else:
        return new_result


def _process_arguments(schema, spec, kwargs, operator_name):
    """
    Process arguments: validate, deprecate and add to spec, handling appropriate data marshalling.
    """
    kwargs = _handle_arg_deprecations(schema, kwargs, operator_name)
    _add_spec_args(schema, spec, kwargs)


def _process_argument_inputs(schema, spec, kwargs, operator_name):
    """Process argument inputs: validate, deprecate, promote constants, add to spec and return the
    list of DataNodes that need to be kept alive for the graph to exist.

    Returns
    -------
    List
        the list of DataNodes representing inputs (there may be conversions)
    """
    kwargs = _handle_arg_deprecations(schema, kwargs, operator_name)
    # Add argument inputs
    result = []
    for k in sorted(kwargs.keys()):
        arg_inp = kwargs[k]
        if arg_inp is None:
            continue
        # Argument input constants are always placed on CPU
        # TODO(klecki): For MIS we can extract this step outside and do it once
        arg_inp = _handle_constant(arg_inp, "cpu", k, operator_name)

        if arg_inp.device != "cpu":
            raise ValueError(
                f"Invalid device \"{arg_inp.device}\" for argument '{k}' of operator "
                f"'{operator_name}'. "
                f"Named arguments must be on CPU."
            )

        _check_arg_input(schema, operator_name, k)

        spec.AddArgumentInput(k, arg_inp.name)
        result.append(arg_inp)
    return result


def _process_inputs(schema, spec, inputs, operator_name):
    """Process inputs: validate, add to spec and return the list of DataNodes that need to be kept
    alive for the graph to exist.

    Returns
    -------
    List
        the list of DataNodes representing inputs (there may be conversions)
    """
    if len(inputs) < schema.MinNumInput() or len(inputs) > schema.MaxNumInput():
        raise ValueError(
            f"Operator {operator_name} expects "
            f"from {schema.MinNumInput()} to {schema.MaxNumInput()} inputs, "
            f"but received {len(inputs)}."
        )
    if not inputs:
        return []
    for inp in inputs:
        if not isinstance(inp, _DataNode):
            raise TypeError(f"Expected inputs of type 'DataNode'. Received input of type '{inp}'.")
        spec.AddInput(inp.name, inp.device)
    return list(inputs)


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
    """Class representing the operator node in DALI Pipeline/graph.
    The entry point accepts grouped arguments: inputs (positional arguments with DataNodes),
    arguments (dictionary with scalar values), and argument inputs (dictionary with DataNodes).

    It is responsible for the full* validation of the three input/argument kinds and their
    additional processing (scalar promotions, deprecations and renaming, splitting for conditionals)

    Each "_process_[arguments/inputs]" step adds them to the OpSpec specifying the operator for
    the C++ backend.

    * some validation is done in class Operator.__init__ due to legacy reasons.
    This is the reason for `_processed_arguments` constructor parameter.
    """

    def __init__(self, inputs, arg_inputs, arguments, _processed_arguments, op):
        """Construct the OperatorInstance and handle the processing of all inputs and arguments.

        Parameters
        ----------
        inputs : tuple of DataNode
            Positional inputs to operator instance.
        arg_inputs : dict of DataNode
            Argument inputs - named DataNode inputs to operator.
        arguments : dict of scalar values
            Scalar arguments to be added to OpSpec.
        _processed_arguments : dict of scalar values
            Remaining scalar arguments. Available here for completeness, for historical reasons
            already validated in Operator.__init__.
            NOTE: they are already added to the `op.spec`!
        op : Operator class.
            Operator class containing the schema, and spec filled with `processed_arguments`.
        """

        if _Pipeline.current():
            self._pipeline = weakref.ref(_Pipeline.current())
        else:
            self._pipeline = None
        self._id = None
        self._outputs = []
        self._op = op
        self._spec = op.spec.copy()
        self._relation_id = None

        if _conditionals.conditionals_enabled():
            inputs, arg_inputs = _conditionals.apply_conditional_split_to_args(inputs, arg_inputs)
            _conditionals.inject_implicit_scope_argument(op._schema, arg_inputs)

        self._process_instance_name(arguments)
        self._process_trace(arguments)
        self._spec.AddArg("preserve_name", not self._autoname)
        _process_arguments(op._schema, self._spec, arguments, op._operator_name())

        self._inputs = _process_inputs(op._schema, self._spec, inputs, op._operator_name())
        self._inputs += _process_argument_inputs(
            op._schema, self._spec, arg_inputs, op._operator_name()
        )

        _handle_op_deprecation(
            self._op.schema, _processed_arguments["_module"], _processed_arguments["_display_name"]
        )

        self._generate_outputs()

        if _conditionals.conditionals_enabled():
            _conditionals.register_data_nodes(self.outputs, inputs, arg_inputs)

    def check_args(self):
        self._op.schema.CheckArgs(self._spec)

    def _process_instance_name(self, arguments):
        """Extract the name from the arguments or generate the default one.
        The "name" is removed from arguments.

        Parameters
        ----------
        arguments : dict
            Dictionary of all scalar arguments passed to the operator
        """
        name = arguments.pop("name", None)
        if name is not None:
            self._name = name
            self._autoname = False
        else:
            has_pipeline = self.pipeline is not None
            # to avoid mixing up global and per-pipeline ids
            infix = "_" if has_pipeline else "_detached_"
            self._name = "__" + type(self._op).__name__ + infix + str(self.id)
            self._autoname = True

    def _process_trace(self, arguments):
        from nvidia.dali._debug_mode import _PipelineDebug

        current_pipeline = _PipelineDebug.current()
        is_debug = getattr(current_pipeline, "_debug_on", False)
        if _dali_trace.is_tracing_enabled() and not is_debug:
            if _Pipeline.current():
                start_frame = _Pipeline.current()._definition_frame_start
            else:
                start_frame = 0
            end_frame = self._op._definition_frame_end
            stack_summary = _dali_trace.extract_stack(start_frame=start_frame, end_frame=end_frame)
            filenames, linenos, names, lines = _dali_trace.preprocess_stack_summary(stack_summary)

            arguments["_origin_stack_filename"] = filenames
            arguments["_origin_stack_lineno"] = linenos
            arguments["_origin_stack_name"] = names
            arguments["_origin_stack_line"] = lines

    def _generate_outputs(self):
        pipeline = _Pipeline.current()
        if pipeline is None and self._op.preserve:
            _Pipeline._raise_pipeline_required("Operators with side-effects ")
        # Add outputs
        if self._op.device == "gpu" or self._op.device == "mixed":
            output_device = "gpu"
        else:
            output_device = "cpu"

        num_output = self._op.schema.CalculateOutputs(
            self._spec
        ) + self._op.schema.CalculateAdditionalOutputs(self._spec)

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
    def pipeline(self):
        return None if self._pipeline is None else self._pipeline()

    @property
    def id(self):
        if self._id is None:
            if self.pipeline is None and _Pipeline.current():
                self._pipeline = weakref.ref(_Pipeline.current())
            if self.pipeline:
                self._id = self.pipeline._next_op_id()
            else:
                self._id = _OpCounter().id

        return self._id

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
        if self._relation_id is None:
            self._relation_id = id(self)
        return self._relation_id

    @relation_id.setter
    def relation_id(self, value):
        self._relation_id = value

    def append_output(self, output):
        self._outputs.append(output)


class _DaliOperatorMeta(type):
    @property
    def __doc__(self):
        return _docs._docstring_generator(_names._schema_name(self))


def _check_arg_input(schema, op_name, name):
    if not schema.IsTensorArgument(name):
        expected_type_name = _type_name_convert_to_string(schema.GetArgumentType(name), False)
        raise TypeError(
            f"The argument `{name}` for operator `{op_name}` should not be a 'DataNode' but a "
            f"'{expected_type_name}'."
        )


def python_op_factory(name, schema_name, internal_schema_name=None, generated=True):
    """Generate the ops API class bindings for operator.

    Parameters
    ----------
    name : str
        The name of the operator (without the module) - this will be the name of the class
    schema_name : str
        Name of the schema, used for documentation lookups and schema/spec retrieval unless
        internal_schema_name  is provided
    internal_schema_name : str, optional
        If provided, this will be the schema used to process the arguments, by default None
    generated : bool, optional
        Mark this class as fully generated API binding (True), or as a (base) class used for
        manually extending the binding code (False), by default True.
    """

    class Operator(metaclass=_DaliOperatorMeta):
        def __init__(self, *, device="cpu", **kwargs):
            if self._internal_schema_name is None:
                schema_name = _schema_name(type(self))
            else:
                schema_name = self._internal_schema_name
            self._spec = _b.OpSpec(schema_name)
            self._schema = _b.GetSchema(schema_name)

            # Get the device argument. We will need this to determine the device that our outputs
            # will be stored on. The argument is not listed in schema, so we need to add it
            # manually to spec. TODO(klecki): Make it more generic.
            self._device = device
            self._spec.AddArg("device", self._device)

            self._init_args, self._call_args = _separate_kwargs(kwargs)

            # Capture the ops API display name if it didn't arrive from fn API.
            if "_module" not in self._init_args:
                self._init_args.update({"_module": Operator.__module__.replace(".hidden", "")})
            if "_display_name" not in self._init_args:
                self._init_args.update({"_display_name": type(self).__name__})

            operator_name = self._operator_name()

            for k in self._call_args.keys():
                _check_arg_input(self._schema, operator_name, k)

            self._preserve = self._init_args.get("preserve", False)

            # Process the first part of arguments, due to the historical reasons
            # we need to do it in __init__ for error reporting
            # TODO(klecki): Get rid of two-stage adding of scalar arguments - we can do it
            # but the error message would be worse or delayed.
            # Name is handled in the op instance, keep it for later.
            self._name = self._init_args.pop("name", None)
            # Stack frame is also processed by the operator instance and we need to remove it
            # before it is validated against Schema.
            if _dali_trace.is_tracing_enabled():
                self._definition_frame_end = self._init_args.pop("_definition_frame_end", None)
            # Make sure that the internal name arguments are added first in case backend
            # needs them to report errors.
            name_internal_keys = ["_display_name", "_module"]
            name_args = {key: self._init_args.pop(key) for key in name_internal_keys}
            _process_arguments(self._schema, self._spec, name_args, operator_name)
            _process_arguments(self._schema, self._spec, self._init_args, operator_name)
            self._init_args.update(name_args)

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
            inputs = _preprocess_inputs(inputs, self._operator_name(), self._device, self._schema)
            input_sets = _build_input_sets(inputs, self._operator_name())

            args, arg_inputs = _separate_kwargs(kwargs)

            # Due to the fact that we already handled *some* args in init, we need to keep only
            # the new ones.
            args = _resolve_double_definitions(args, self._init_args, keep_old=False)
            if self._name is not None:
                args = _resolve_double_definitions(args, {"name": self._name})  # restore the name

            if _dali_trace.is_tracing_enabled() and self._definition_frame_end is None:
                self._definition_frame_end = _dali_trace.get_stack_depth() - 1

            self._preserve = (
                self._preserve or args.get("preserve", False) or self._schema.IsNoPrune()
            )

            # Adding argument inputs is fully delayed into call, so we just do the check
            arg_inputs = _resolve_double_definitions(arg_inputs, self._call_args)

            # Create OperatorInstance for every input set.
            # OperatorInstance handles the creation of OpSpec and generation of output DataNodes
            op_instances = []
            for input_set in input_sets:
                op_instances.append(
                    _OperatorInstance(input_set, arg_inputs, args, self._init_args, self)
                )

            # Tie the instances together
            relation_id = op_instances[0].relation_id
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
            return result

        def _operator_name(self):
            """
            Return a proper display name of operator based on the API it was instantiated in.

            Only valid after `__init__` kwargs were split into `_init_args` and `_call_args`.
            """
            return f"{self._init_args['_module']}.{self._init_args['_display_name']}"

    Operator.__name__ = str(name)
    Operator.schema_name = schema_name
    Operator._internal_schema_name = internal_schema_name
    Operator._generated = generated
    Operator.__call__.__doc__ = _docs._docstring_generator_call(Operator.schema_name)
    if _b.TryGetSchema(schema_name) is not None:
        schema = _b.GetSchema(schema_name)
        from nvidia.dali.ops import _signatures

        Operator.__init__.__signature__ = _signatures._call_signature(
            schema,
            include_inputs=False,
            include_kwargs=True,
            include_self=True,
            data_node_return=False,
            all_args_optional=True,
        )
        Operator.__call__.__signature__ = _signatures._call_signature(
            schema,
            include_inputs=True,
            include_kwargs=True,
            include_self=True,
            all_args_optional=True,
        )
    return Operator


def _wrap_op(op_class, submodule=[], parent_module=None):
    return _functional._wrap_op(
        op_class, submodule, parent_module, _docs._docstring_generator_fn(op_class.schema_name)
    )


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
            _internal._adjust_operator_module(op_class, ops_module, submodule)
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
    """After backend ops are loaded, load the TFRecord readers (if they are available)."""
    from nvidia.dali.ops._operators import tfrecord

    if not tfrecord.tfrecord_enabled():
        return

    _registry.register_cpu_op("readers__TFRecord")
    _registry.register_cpu_op("TFRecordReader")

    ops_module = sys.modules[__name__]

    for op_reg_name, op_class in [
        ("readers__TFRecord", tfrecord.TFRecord),
        ("TFRecordReader", tfrecord.TFRecordReader),
    ]:
        _, submodule, op_name = _process_op_name(op_reg_name)
        module = _internal.get_submodule(ops_module, submodule)
        if not hasattr(module, op_name):
            _internal._adjust_operator_module(op_class, ops_module, submodule)
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

    if schema and (len(inputs) < schema.MinNumInput() or len(inputs) > schema.MaxNumInput()):
        raise ValueError(
            f"Operator {op_name} expects "
            f"from {schema.MinNumInput()} to {schema.MaxNumInput()} inputs, "
            f"but received {len(inputs)}."
        )

    def is_input(x):
        if isinstance(x, (_DataNode, nvidia.dali.types.ScalarConstant)):
            return True
        # One level of nesting for Multiple Input Sets. It must be a List[DataNode/ScalarConstant]
        # with at least one DataNode.
        return (
            isinstance(x, (list))
            and any(isinstance(y, _DataNode) for y in x)
            and all(isinstance(y, (_DataNode, nvidia.dali.types.ScalarConstant)) for y in x)
        )

    def get_input_device(schema, input_idx):
        default_input_device = "gpu" if device == "gpu" else "cpu"
        if schema:
            input_device = schema.GetInputDevice(input_idx, None, default_input_device)
        else:
            input_device = default_input_device
        return input_device or default_input_device

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
                raise TypeError(
                    f"when calling operator `{op_name}`: "
                    f"expected inputs of type 'DataNode', list of 'DataNode' "
                    f"or convertible to constant nodes. Received "
                    f"input `{idx}` of type '{type(inp).__name__}'."
                ) from ex

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

_internal._adjust_operator_module(ExternalSource, sys.modules[__name__], [])

# Expose the PythonFunction family of classes and generate the fn bindings for them
from nvidia.dali.ops._operators.python_function import (  # noqa: E402, F401
    PythonFunction,
    DLTensorPythonFunction,
    _dlpack_to_array,  # noqa: F401
    _dlpack_from_array,
)  # noqa: F401

_internal._adjust_operator_module(PythonFunction, sys.modules[__name__], [])
_internal._adjust_operator_module(DLTensorPythonFunction, sys.modules[__name__], [])

_wrap_op(PythonFunction)
_wrap_op(DLTensorPythonFunction)

# Compose is only exposed for ops API, no fn bindings are generated
from nvidia.dali.ops._operators.compose import Compose as Compose  # noqa: E402, F401

_internal._adjust_operator_module(Compose, sys.modules[__name__], [])


from nvidia.dali.ops._operators.math import (  # noqa: F401, E402
    _arithm_op,
    _group_inputs,
    _generate_input_desc,
)


# Discover and generate bindings for all regular operators.
_load_ops()

# Load the TFRecord after the backend ops are processed, to wrap it conditionally, if it exists.
_load_readers_tfrecord()
