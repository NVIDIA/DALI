# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import builtins
import os
import re
import string
import tokenize
from contextlib import closing
from inspect import Parameter, Signature, getdoc, getmodule, ismodule
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

from nvidia.dali import backend as _b
from nvidia.dali import fn, ops, types
from nvidia.dali import types as _types
from nvidia.dali.ops import _docs, _names, _registry

Api = Literal["fn", "ops", "dynamic"]


def _create_annotation_placeholder(typename):
    # Inspect and typing know better how to format annotations. They produce short representations
    # of types that are builtins or belonging to typing module, but anything else gets a fully
    # qualified path and there is no customization. Also the formatting using the `repr` of the type
    # is driven by inspect.formatannotation (when we use the type directly) or by the internals of
    # typing classes like Union. So we just pretend, that we are from typing, and we want to
    # have our name as `typename`.

    class _AnnotationPlaceholderMeta(type):
        # We don't want <class 'typename'> when inspect calls inspect.formatannotation directly.
        def __repr__(cls):
            return typename

    class _AnnotationPlaceholder(metaclass=_AnnotationPlaceholderMeta):
        # typing uses those instead of repr
        __name__ = typename
        __qualname__ = typename
        # only if you are part of `typing.` your path is hidden
        __module__ = "typing"

        def __init__(self, val):
            self._val = val

        # Work around the repr of enums, by replacing them with `__str__`
        def __repr__(self):
            return str(self._val)

    return _AnnotationPlaceholder


# This is not the DataNode you are looking for.
_DataNode = _create_annotation_placeholder("DataNode")

# The placeholder for the DALI Enum types, as the bindings from backend don't play nice,
# we need actual Python classes.
_DALIDataType = _create_annotation_placeholder("DALIDataType")
_DALIImageType = _create_annotation_placeholder("DALIImageType")
_DALIInterpType = _create_annotation_placeholder("DALIInterpType")
_TensorLikeIn = _create_annotation_placeholder("TensorLikeIn")
_TensorLikeArg = _create_annotation_placeholder("TensorLikeArg")
_RNG = _create_annotation_placeholder("RNG")

_enum_mapping = {
    types.DALIDataType: _DALIDataType,
    types.DALIImageType: _DALIImageType,
    types.DALIInterpType: _DALIInterpType,
}

# Placeholders for dynamic mode
_Tensor = _create_annotation_placeholder("Tensor")
_Batch = _create_annotation_placeholder("Batch")
_TensorLike = _create_annotation_placeholder("TensorLike")
_DType = _create_annotation_placeholder("DType")


def _api_to_module(api: Api):
    if api == "fn":
        return fn
    if api == "ops":
        return ops
    if api == "dynamic":
        from nvidia.dali.experimental import dynamic

        return dynamic


def _scalar_element_annotation(scalar_dtype, api: Api):
    # We already have function that converts a scalar constant/literal into the desired type,
    # utilize the fact that they accept integer values and get the actual type.
    conv_fn = _types._known_types[scalar_dtype][1]
    try:
        dummy_val = conv_fn(0)
        t = type(dummy_val)

        if api == "dynamic" and t is types.DALIDataType:
            t = Union[_DALIDataType, _DType]

        if t in _enum_mapping:
            return _enum_mapping[t]

        api_module = _api_to_module(api)
        if (
            hasattr(t, "__name__")
            and hasattr(api_module, t.__name__)
            and hasattr(builtins, t.__name__)
        ):
            # Resolve conflicts between exported symbols and types used in annotations
            # For instance, bool becomes "builtins.bool" because of ndd.bool
            t = f"builtins.{t.__name__}"  # type: ignore[reportAttributeAccessIssue]
        return t
    # This is tied to TFRecord implementation
    except NotImplementedError:
        return Any
    except TypeError:
        return Any


def _arg_type_annotation(arg_dtype, api: Api):
    """Convert regular key-word argument type to annotation. Handles Lists and scalars.

    Parameters
    ----------
    arg_dtype : DALIDataType
        The type information from schema
    """
    if arg_dtype in _types._vector_types:
        scalar_dtype = _types._vector_types[arg_dtype]
        scalar_annotation = _scalar_element_annotation(scalar_dtype, api)
        # DALI allows tuples and lists as a "sequence" parameter
        return Union[Sequence[scalar_annotation], scalar_annotation]
    return _scalar_element_annotation(arg_dtype, api)


def _get_positional_input_param(schema, idx, annotation):
    """Get the Parameter representing positional inputs at `idx`. Automatically mark it as
    optional. The DataNode annotation currently hides the possibility of MIS.

    The double underscore `__` prefix for argument name is an additional way to indicate
    positional only arguments, as per MyPy docs. It is obeyed by the VSCode.
    """
    # Only first MinNumInputs are mandatory, the rest are optional:
    default = Parameter.empty if idx < schema.MinNumInput() else None
    annotation = annotation if idx < schema.MinNumInput() else Optional[annotation]
    return Parameter(
        _names._get_input_name(schema, idx),
        kind=Parameter.POSITIONAL_ONLY,
        default=default,
        annotation=annotation,
    )


def _get_annotation_input_regular(schema):
    """Return the annotation for regular input parameter in DALI, used for the primary overload.
    A function is used as a global variable can be confused with type alias.
    """
    return Union[_DataNode, _TensorLikeIn]


def _get_annotation_return_regular(schema):
    """Produce the return annotation for DALI operator suitable for primary, non-MIS overload.
    Note the flattening, single output is not packed in Sequence.
    """
    if schema.HasOutputFn():
        # Dynamic number of outputs, not known at "compile time"
        return_annotation = Union[_DataNode, Sequence[_DataNode], None]
    else:
        # Call it with a dummy spec, as we don't have Output function
        num_regular_output = schema.CalculateOutputs(_b.OpSpec(""))
        if num_regular_output == 0:
            return_annotation = None
        elif num_regular_output == 1:
            return_annotation = _DataNode
        else:
            # Here we could utilize the fact, that the tuple has known length, but we can't
            # as DALI operators return a list
            # Also, we don't advertise the actual List type, hence the Sequence.
            return_annotation = Sequence[_DataNode]
    return return_annotation


def _get_annotation_input_mis(schema):
    """Return the annotation for multiple input sets, used for the secondary operator overload.
    A function is used as a global variable can be confused with type alias.
    Handles special case for operators with one input.
    """
    if schema.MinNumInput() == 1 and schema.MaxNumInput() == 1:
        return List[_DataNode]
    return Union[List[_DataNode], _DataNode, _TensorLikeIn]


def _get_annotation_return_mis(schema):
    """Annotation for function that handles Multiple input sets overload.
    Note that DALI does a lot of flattening, so single-element sequences are transformed to
    just that element.
    """
    if schema.HasOutputFn():
        # Dynamic number of outputs, not known at "compile time"
        # We can return single or multiple outputs in regular case (same as primary overload),
        # a list of single outputs or a list of multiple outputs for MIS or None.
        return_annotation = Union[
            _DataNode, Sequence[_DataNode], List[_DataNode], List[Sequence[_DataNode]], None
        ]
    else:
        # Call it with a dummy spec, as we don't have Output function
        num_regular_output = schema.CalculateOutputs(_b.OpSpec(""))
        if num_regular_output == 0:
            return_annotation = None
        elif num_regular_output == 1:
            # This allows for type hints with single-return operators to work correctly with MIS
            # as the return type matches the overload for their input type.
            return_annotation = Union[_DataNode, List[_DataNode]]
        else:
            # Here we could utilize the fact, that the tuple has known length, but we can't
            # as DALI operators return a list
            # Also, we don't advertise the actual List type, hence the Sequence, but we say
            # that the outermost return type of MIS is a List.
            return_annotation = Union[Sequence[_DataNode], List[Sequence[_DataNode]]]
    return return_annotation


def _get_positional_input_params(schema, input_annotation_gen=_get_annotation_input_regular):
    """Get the list of positional only inputs to the operator.

    Parameters
    ----------
    input_annotation_gen: Callable[[OpSchema], type annotation]
        Input type annotation, used to indicate regular inputs or multiple input set overloads.
        See _get_annotation_* functions.
    """
    param_list = []
    # If outputs are documented, list all of them
    if schema.HasInputDox():
        for i in range(schema.MaxNumInput()):
            param_list.append(
                _get_positional_input_param(schema, i, annotation=input_annotation_gen(schema))
            )
    else:
        # List all mandatory inputs
        for i in range(schema.MinNumInput()):
            param_list.append(
                _get_positional_input_param(schema, i, annotation=input_annotation_gen(schema))
            )
        # If they fit below limit, list all inputs (with optional ones)
        if schema.MaxNumInput() < _docs._MAX_INPUT_SPELLED_OUT:
            for i in range(schema.MinNumInput(), schema.MaxNumInput()):
                param_list.append(
                    _get_positional_input_param(schema, i, annotation=input_annotation_gen(schema))
                )
        # List the rest of optional inputs in general fashion
        elif schema.MaxNumInput() > schema.MinNumInput():
            # Note that the VAR_POSTIONAL annotation means that all arguments passed this way
            # have to conform to this type, it doesn't get a default None value as it already is
            # "empty" by default - def(*args = None) is invalid syntax.
            param_list.append(
                Parameter(
                    _names._get_variadic_input_name(),
                    Parameter.VAR_POSITIONAL,
                    annotation=Optional[input_annotation_gen(schema)],
                )
            )
    return param_list


def _get_keyword_params(
    schema,
    api: Api,
    all_args_optional: bool,
    data_node_tensors: bool,
    include_kwarg_inputs: bool,
    include_only_inputs: bool,
):
    """Get the list of annotated keyword Parameters to the operator."""

    if api == "dynamic":
        from nvidia.dali.experimental.dynamic._op_builder import _unsupported_args

    param_list = []
    for arg in schema.GetArgumentNames():
        if api == "dynamic" and arg in _unsupported_args:  # type: ignore
            continue

        if schema.IsDeprecatedArg(arg):
            # We don't put the deprecated args in the visible API
            continue
        arg_dtype = schema.GetArgumentType(arg)
        kw_annotation = _arg_type_annotation(arg_dtype, api)
        is_arg_input = schema.IsTensorArgument(arg)

        if not is_arg_input and include_only_inputs:
            continue

        if is_arg_input and not include_kwarg_inputs:
            continue

        if is_arg_input and include_kwarg_inputs:
            annotation = (
                Union[_DataNode, _TensorLikeArg, kw_annotation]
                if data_node_tensors
                else Union[_TensorLikeArg, _Batch, kw_annotation]
            )
        else:
            annotation = kw_annotation

        if schema.IsArgumentOptional(arg):
            # In DALI arguments can always accept optional, and the propagation of such argument
            # is skipped. Passing None is equivalent to not providing the argument at all,
            # and is typically resolved to the default value (but can have special meaning).
            annotation = Optional[annotation]

        default = Parameter.empty
        if schema.HasArgumentDefaultValue(arg):
            default_value_string = schema.GetArgumentDefaultValueString(arg)
            default_value = ast.literal_eval(default_value_string)
            default = types._type_convert_value(arg_dtype, default_value)
            if type(default) in _enum_mapping:
                default = _enum_mapping[type(default)](default)
        elif schema.IsArgumentOptional(arg):
            default = None

        # Workaround for `ops` API where keyword args can be specified in either one of __init__
        # and __call__, so we can't make them mandatory anywhere.
        if all_args_optional:
            annotation = Optional[annotation]
            if default == Parameter.empty:
                default = None

        param_list.append(
            Parameter(name=arg, kind=Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )

    # We omit the **kwargs, as we already specified all possible parameters:
    # param_list.append(Parameter("kwargs", Parameter.VAR_KEYWORD))
    # We could add it, but it would behave as catch all.
    return param_list


def _get_implicit_extra_params(schema, api: Api, include_init_header: bool):
    """All operators have some parameters, that are not listed in schema, but are
    implicitly used by DALI.
    If include_init_header is True, arguments are positional or keyword, so the order matters.
    """

    supported_backends = schema.GetSupportedBackends()
    if api == "dynamic" and "mixed" in supported_backends:
        supported_backends.append("gpu")

    device_annotation = Literal[tuple(supported_backends)] if supported_backends else str

    if include_init_header:
        params = [
            Parameter(
                name="max_batch_size",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[int],
            ),
            Parameter(
                name="name",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[str],
            ),
            Parameter(
                name="device",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default="cpu",
                annotation=Union["Device", device_annotation],  # noqa # type: ignore
            ),
            Parameter(
                name="num_inputs",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[int],
            ),
        ]
    else:
        params = [
            # TODO(klecki): The default for `device` is dependant on the input placement (and API).
            Parameter(
                name="device",
                kind=Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[device_annotation],
            )
        ]
        if api != "dynamic":
            # The name is truly optional
            params.append(
                Parameter(
                    name="name",
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Optional[str],
                )
            )
        elif "random" in schema.ModulePath():
            params.append(
                Parameter(
                    name="rng",
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Optional[_RNG],
                )
            )

    return params


def _call_signature(
    schema,
    api: Api,
    include_inputs=True,
    include_kwarg_inputs=True,
    include_only_inputs=False,
    include_kwargs=True,
    include_self=False,
    include_batch_size=False,
    return_annotation=True,
    data_node_kwargs=True,
    all_args_optional=False,
    include_init_header=False,
    input_annotation_gen=_get_annotation_input_regular,
    return_annotation_gen=_get_annotation_return_regular,
    filter_annotations=False,
) -> Signature:
    """Generate a Signature for given schema.

    Parameters
    ----------
    schema : OpSchema
        Schema for the operator.
    api : str
        "fn", "ops" or "dynamic"
    include_inputs : bool, optional
        If positional inputs should be included in the signature, by default True
    include_kwarg_inputs: bool, optional
        If inputs that are keyword arguments should be included in the signature, by default True
    include_only_inputs: bool, optional
        If non-input keyword arguments should be excluded, by default False
    include_kwargs : bool, optional
        If keyword arguments should be included in the signature, by default True
    include_self : bool, optional
        Prepend `self` as first positional argument in the signature, by default False
    include_batch_size : bool, optional
        Prepend `batch_size` as first keyword-only argument in the signature, by default False
    return_annotation : bool, optional
        If the signature should have a return annotation or return None (for ops class __init__),
        by default True
    data_node_kwargs : bool, optional
        If tensor keyword arguments should accept DataNodes, by default True
    all_args_optional : bool, optional
        Make all keyword arguments optional, even if they are not - needed by the ops API, where
        the argument can be specified in either __init__ or __call__, by default False
    include_init_header : bool, optional
        Ignored if api != 'dynamic'. Used to include extra arguments in __init__, by default False
    input_annotation_gen : Callable[[OpSchema], type annotation]
        Callback generating the annotation to be used for type annotation of inputs.
    return_annotation_gen : Callable[[OpSchema], type annotation]
        Callback generating the return type annotation for given schema
    """
    param_list = []
    if include_self:
        param_list.append(Parameter("self", kind=Parameter.POSITIONAL_ONLY))

    if include_inputs:
        param_list.extend(
            _get_positional_input_params(schema, input_annotation_gen=input_annotation_gen)
        )

    if include_kwargs and not include_only_inputs:
        include_init_header = include_init_header and api == "dynamic"
        param_list.extend(_get_implicit_extra_params(schema, api, include_init_header))

    if include_batch_size:
        param_list.append(Parameter(name="batch_size", kind=Parameter.KEYWORD_ONLY, annotation=int))

    if include_kwargs:
        param_list.extend(
            _get_keyword_params(
                schema,
                api,
                all_args_optional=all_args_optional,
                data_node_tensors=data_node_kwargs,
                include_kwarg_inputs=include_kwarg_inputs,
                include_only_inputs=include_only_inputs,
            )
        )

    if return_annotation:
        return_annotation = return_annotation_gen(schema)
    else:
        return_annotation = None
    if filter_annotations:
        param_list = [Parameter(name=p.name, kind=p.kind, default=p.default) for p in param_list]
        return_annotation = Signature.empty
    return Signature(param_list, return_annotation=return_annotation)


def inspect_repr_fixups(signature: str) -> str:
    """Replace the weird quirks of printing the repr of signature.
    We use signature object for type safety and additional validation, but the printing rules
    are questionable in some cases:
    - Python type hints advocate the usage of `None` instead of its type, but printing a signature
      would insert NoneType (specifically replacing Optional[Union[...]] with Union[..., None] and
      printing it as Union[..., NoneType]). The NoneType doesn't exist as a `types` definition in
      some Pythons versions.
    - Optional["SomeClass"] get translated to Optional[ForwardRef("SomeClass")], which then can
      confuse type checkers. Remove forward refs
    """

    return re.sub(r"ForwardRef\('([^']+)'\)", r"\1", signature).replace("NoneType", "None")


def _gen_fn_signature_no_input(schema, schema_name, fn_name):
    """In case of no input, we don't have overload set, as there is no MIS involved,
    we write only the default signature, without involving @overload decorator.
    """
    return f"""
def {fn_name}{_call_signature(schema, "fn", include_inputs=True, include_kwargs=True)}:
    \"""{_docs._docstring_generator_fn(schema_name)}
    \"""
    ...
"""


def _gen_fn_signature_with_inputs(schema, schema_name, fn_name):
    """Generate primary and secondary overload (for regular and MIS cases).

    Python resolves the overloads in order of definition, we will match first against the primary
    overload accepting only the DataNode, and if any of the inputs is a list of such (indicating
    Multiple Input Sets), we will match the second overload that is more general.
    The secondary overload has less constrained return type annotation but we have to accept it
    to not have exponential number of overloads.
    There is a special case, where the secondary overload with exactly one input accepts only
    List[DataNode] (see the `_get_annotation_input_mis`). Passing a variable recognized as
    Union[DataNode, List[DataNode]] to such overload set still resolves correctly under mypy and
    pylance, resulting again in the Union[DataNode, List[DataNode]] return type (for single output),
    keeping us in the MIS realm in the simple case.
    """
    return f"""
@overload
def {fn_name}{_call_signature(schema, "fn", include_inputs=True, include_kwargs=True)}:
    \"""{_docs._docstring_generator_fn(schema_name)}
    \"""
    ...


@overload
def {fn_name}{_call_signature(schema, "fn", include_inputs=True, include_kwargs=True,
                              input_annotation_gen=_get_annotation_input_mis,
                              return_annotation_gen=_get_annotation_return_mis)}:
    \"""{_docs._docstring_generator_fn(schema_name)}
    \"""
    ...
"""


def _gen_fn_signature(schema, schema_name, fn_name):
    """Write the stub of the fn API function with the docstring, for given operator.
    Include two overloads: with regular inputs and secondary accepting MIS.
    If there are no inputs, we have only one signature.
    """
    if schema.MaxNumInput() == 0:
        return inspect_repr_fixups(_gen_fn_signature_no_input(schema, schema_name, fn_name))
    else:
        return inspect_repr_fixups(_gen_fn_signature_with_inputs(schema, schema_name, fn_name))


def _gen_ops_call_signature_no_input(schema, schema_name):
    """In case of no input, we don't have overload set, as there is no MIS involved,
    we write only the default call signature, without involving @overload decorator.
    """
    return f"""
    def __call__{_call_signature(schema, "ops", include_inputs=True, include_kwargs=True,
                                 include_self=True, all_args_optional=True)}:
        \"""{_docs._docstring_generator_call(schema_name)}
        \"""
        ...
"""


def _gen_ops_call_signature_with_inputs(schema, schema_name):
    """Generate primary and secondary overload (for regular and MIS cases).
    Read _gen_fn_signature_with_inputs docstring for details - this is the same thing for ops API.
    """
    signature = _call_signature(
        schema,
        "ops",
        include_inputs=True,
        include_kwargs=True,
        include_self=True,
        all_args_optional=True,
        input_annotation_gen=_get_annotation_input_mis,
        return_annotation_gen=_get_annotation_return_mis,
    )
    return f"""
    @overload
    def __call__{_call_signature(schema, "ops", include_inputs=True, include_kwargs=True,
                                 include_self=True, all_args_optional=True)}:
        \"""{_docs._docstring_generator_call(schema_name)}
        \"""
        ...

    @overload
    def __call__{signature}:
        \"""{_docs._docstring_generator_call(schema_name)}
        \"""
        ...
"""


def _gen_ops_signature(schema, schema_name, cls_name):
    """Write the stub of the fn API class with the docstring, __init__ and __call__ for given
    operator.
    """
    signature = (
        _gen_ops_call_signature_no_input(schema, schema_name)
        if schema.MaxNumInput() == 0
        else _gen_ops_call_signature_with_inputs(schema, schema_name)
    )
    return inspect_repr_fixups(
        f"""
class {cls_name}:
    \"""{_docs._docstring_generator_class(schema_name)}
    \"""
    def __init__{_call_signature(schema, "ops", include_inputs=False, include_kwargs=True,
                                 include_self=True, return_annotation=False,
                                 all_args_optional=True)}:
        ...

{signature}
"""
    )


def _gen_dynamic_call_signature_no_input(schema: _b.OpSchema, **kwargs):
    """Generate function signatures for no-input dynamic mode ops. The overloads are:
    - `(**kwargs) -> tensor-like`:
      Calling a no-input parameter without specifying a batch size returns a single sample
    - `(*, batch_size: int, **kwargs) -> batch`
      Invocation with a batch size returns a batch.
    """
    yield from (
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            return_annotation_gen=lambda _: _Tensor,
            **kwargs,
        ),
        _call_signature(
            schema,
            api="dynamic",
            include_batch_size=True,
            data_node_kwargs=False,
            return_annotation_gen=lambda _: _Batch,
            **kwargs,
        ),
    )


def _gen_dynamic_call_signature_single_input(schema: _b.OpSchema, **kwargs):
    """Generate function signatures for single-input dynamic mode ops. The overloads are:
    - `(tensor-like, /, **kwargs) -> Tensor | Batch`:
        When the input is a tensor, it is possible that one or more arguments are batches,
        therefore producing a batch by broadcasting.
    - `(tensor-like, /, *, batch_size: int, **kwargs) -> Batch`:
        If `batch_size` is specified, the output is always a batch.
    - `(batch, /, **kwargs) -> Batch`:
        If the input is a batch, if the `batch_size` argument is set to an integer,
        it either matches and has no effect, or doesn't and causes a runtime error.
        It is therefore ommitted from this overload.
    """
    yield from (
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            input_annotation_gen=lambda _: _TensorLike,
            return_annotation_gen=lambda _: Union[_Tensor, _Batch],
            **kwargs,
        ),
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            include_batch_size=True,
            input_annotation_gen=lambda _: _TensorLike,
            return_annotation_gen=lambda _: _Batch,
            **kwargs,
        ),
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            input_annotation_gen=lambda _: _Batch,
            return_annotation_gen=lambda _: _Batch,
            **kwargs,
        ),
    )


def _gen_dynamic_call_signature_multiple_inputs(schema: _b.OpSchema, **kwargs):
    """Generate function signatures for single-input dynamic mode ops. The logic is similar to
    ``_gen_dynamic_call_signature_single_input`` but functions accept ``TensorLike | Batch``
    instead of ``TensorLike``. Since ``Batch`` <: ``TensorLike | Batch``, signatures are reordered.
    """
    yield from (
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            input_annotation_gen=lambda _: _Batch,
            return_annotation_gen=lambda _: _Batch,
            **kwargs,
        ),
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            input_annotation_gen=lambda _: Union[_TensorLike, _Batch],
            return_annotation_gen=lambda _: Union[_Tensor, _Batch],
            **kwargs,
        ),
        _call_signature(
            schema,
            api="dynamic",
            data_node_kwargs=False,
            include_batch_size=True,
            input_annotation_gen=lambda _: Union[_TensorLike, _Batch],
            return_annotation_gen=lambda _: _Batch,
            **kwargs,
        ),
    )


def _gen_dynamic_call_signature(schema: _b.OpSchema, **kwargs):
    """Generate function signatures for a dynamic mode ops. The values produced by this function
    can be used to fill the parameters of the function or __call__ method of the operator.
    Additional arguments can will be forwarded to _call_signature
    """

    num_inputs = schema.MaxNumInput()
    if num_inputs == 0:
        generator = _gen_dynamic_call_signature_no_input
    elif num_inputs == 1:
        generator = _gen_dynamic_call_signature_single_input
    else:
        generator = _gen_dynamic_call_signature_multiple_inputs

    yield from generator(schema, **kwargs)


def _try_extend_reader_signature(schema: _b.OpSchema, op_name: str):
    """If the operator is a reader, add the signature for ``next_epoch``.
    Return an empty string if not a reader.
    """

    from nvidia.dali.experimental import dynamic

    readers = getattr(dynamic, "readers", None)
    if readers is None:
        return ""
    op = getattr(readers, op_name, None)
    if op is None or not issubclass(op, dynamic.ops.Reader):
        return ""

    doc = getdoc(op)
    return f"""
    @overload
    def next_epoch(
        self,
        ctx: Optional[EvalContext] = None,
    ) -> Union[Iterable[tuple[Tensor, ...]], Iterable[tuple[Batch, ...]]]:
        \"""{doc}
        \"""

    @overload
    def next_epoch(
        self,
        batch_size: int, ctx: Optional[EvalContext] = None,
    ) -> Iterable[tuple[Batch, ...]]:
        \"""{doc}
        \"""
"""


def _gen_dynamic_cls_signature(schema: _b.OpSchema, schema_name: str, op_name: str):
    call_template = string.Template(
        """
    @overload
    def __call__$signature:
        ...
    """
    )
    call_overloads = (
        call_template.substitute(signature=signature)
        for signature in _gen_dynamic_call_signature(
            schema,
            include_self=True,
            include_only_inputs=True,
        )
    )

    return f"""
class {op_name}:
    \"""{_docs._docstring_generator_class(schema_name, api="dynamic")}
    \"""

    def __init__{_call_signature(
        schema,
        "dynamic",
        include_inputs=False,
        include_kwarg_inputs=False,
        include_self=True,
        return_annotation=False,
        include_kwargs=True,
        include_init_header=True,
    )}:
        ...

{os.linesep.join(call_overloads)}
{_try_extend_reader_signature(schema, op_name)}
"""


def _gen_dynamic_fun_signature(schema: _b.OpSchema, schema_name: str, op_name: str):
    template = string.Template(
        """
@overload
def $fn_name$signature:
    \"""$doc
    \"""
"""
    )

    doc = _docs._docstring_generator_fn(schema_name, api="dynamic")
    overloads = (
        template.substitute(fn_name=op_name, signature=signature, doc=doc)
        for signature in _gen_dynamic_call_signature(schema)
    )

    return os.linesep.join(overloads)


def _gen_dynamic_signature(schema: _b.OpSchema, schema_name: str, op_name: str):
    """Write the stub of the dynamic API function with the docstring, for given function or class.
    Depending on the number of inputs (0, 1, >1), the number of overloads is different.
    """

    # Determine if we have a class or a function by looking at the case
    # This is a bit hacky but should work as functions are snake-cased.
    if op_name[0].islower():
        signature = _gen_dynamic_fun_signature(schema, schema_name, op_name)
    else:
        signature = _gen_dynamic_cls_signature(schema, schema_name, op_name)

    return inspect_repr_fixups(signature)


# Preamble with license and helper imports for the stub file.
# We need the placeholders for actual Python classes, as the ones that are exported from backend
# don't seem to work with the intellisense.
_HEADER = """
# Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import builtins
from typing import Literal, Optional, Union, overload
from typing import Any, List, Sequence

from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

"""

_PIPELINE_HEADER = """
from nvidia.dali._typing import TensorLikeIn, TensorLikeArg
from nvidia.dali.data_node import DataNode
"""

_DYNAMIC_HEADER = """
from collections.abc import Iterable

from nvidia.dali._typing import TensorLike, TensorLikeArg
from nvidia.dali.experimental.dynamic._batch import Batch as Batch
from nvidia.dali.experimental.dynamic._device import Device as Device
from nvidia.dali.experimental.dynamic._eval_context import EvalContext as EvalContext
from nvidia.dali.experimental.dynamic._tensor import Tensor as Tensor
from nvidia.dali.experimental.dynamic._type import DType as DType
"""


def _build_module_tree():
    """Build a tree of DALI submodules, starting with empty string as a root one, like:
    {
        "" : {
            "decoders" : {},
            "experimental": {
                "readers": {}
            }
            "readers" : {},
        }
    }
    """
    module_tree = {}
    processed = set()
    for schema_name in _registry._all_registered_ops():
        schema = _b.TryGetSchema(schema_name)
        if schema is None:
            continue
        if schema.IsDocHidden() or schema.IsInternal():
            continue
        dotted_name, module_nesting, op_name = _names._process_op_name(schema_name)
        if dotted_name not in processed:
            module_nesting.insert(0, "")  # add the top-level module
            curr_dict = module_tree
            # add all submodules on the path
            for curr_module in module_nesting:
                if curr_module not in curr_dict:
                    curr_dict[curr_module] = dict()
                curr_dict = curr_dict[curr_module]
    return module_tree


def _get_op(api_module, full_qualified_name: List[str]):
    """Resolve the operator function/class from the api_module: ops or fn,
    by accessing the fully qualified name.

    Parameters
    ----------
    api_module : module
        fn or ops
    full_qualified_name : List[str]
        For example ["readers", "File"]
    """
    op = api_module
    for elem in full_qualified_name:
        op = getattr(op, elem, None)
    return op


def _extract_dynamic_mode_definitions():
    """DALI dynamic defines several classes and functions that are not operators and thus
    not exported (e.g., Tensor, as_tensor). Retrieve such definitions.
    """
    from nvidia.dali.experimental import dynamic

    registered_names = set()
    for schema_name in _registry._all_registered_ops():
        *_, op_name = _names._process_op_name(schema_name, api="fn")
        registered_names.add(op_name)

    exported_names = {name for name in dir(dynamic)}
    for name in exported_names - registered_names:
        definition = getattr(dynamic, name)
        module = getmodule(definition)

        # Exclude already re-exported definitions
        if module is None or not module.__name__.startswith(dynamic.__name__):
            continue

        if ismodule(definition):
            module = dynamic

        yield module.__name__, name


def _group_signatures(api: Api):
    """Divide all operators registered into the "ops", "fn" or "dynamic api into 4 categories
    and return them as a dictionary:
    * python_only - there is just the Python definition
    * hidden_or_internal - op is hidden or internal, defined in backend
    * python_wrapper - op defined in backend, has a hand-written wrapper (op._generated = False)
    * generated - op was generated automatically from backend definition (op._generated = True)

    Each entry in the dict contains a list of: `(schema_name : str, op : Callable or Class)`
    depending on the api type.

    """

    sig_groups = {
        "python_only": [],
        "hidden_or_internal": [],
        "python_wrapper": [],
        "generated": [],
    }

    api_module = _api_to_module(api)

    for schema_name in sorted(_registry._all_registered_ops()):
        schema = _b.TryGetSchema(schema_name)

        _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)
        op = _get_op(api_module, module_nesting + [op_name])

        if op is None:
            continue

        if schema is None:
            sig_groups["python_only"].append((schema_name, op))
            continue

        if schema.IsDocHidden() or schema.IsInternal():
            sig_groups["hidden_or_internal"].append((schema_name, op))
            continue

        if not getattr(op, "_generated", False):
            sig_groups["python_wrapper"].append((schema_name, op))
            continue

        sig_groups["generated"].append((schema_name, op))

    return sig_groups


class StubFileManager:
    def __init__(self, nvidia_dali_path: Path, api_path: str):
        self._module_to_file = {}
        self._nvidia_dali_path = nvidia_dali_path
        self._api = api_path
        self._module_tree = _build_module_tree()
        self._header = _HEADER

        if api_path in ("ops", "fn"):
            self._header += _PIPELINE_HEADER
        else:
            self._header += _DYNAMIC_HEADER

    def get(self, module_nesting: List[str]):
        """Get the file representing the given submodule nesting.
        List may be empty for top-level api module.

        When the file is accessed the first time, it's header and submodule imports are
        written.
        """
        module_path = Path("/".join(module_nesting))
        if module_path not in self._module_to_file:
            file_path = self._nvidia_dali_path / self._api / module_path / "__init__.pyi"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, "w").close()  # clear the file
            f = open(file_path, "a")
            self._module_to_file[module_path] = f
            f.write(self._header)
            full_module_nesting = [""] + module_nesting
            # Find out all the direct submodules and add the imports
            submodules_dict = self._module_tree
            for submodule in full_module_nesting:
                submodules_dict = submodules_dict[submodule]
            direct_submodules = submodules_dict.keys()
            for direct_submodule in direct_submodules:
                print(f"from . import {direct_submodule} as {direct_submodule}", file=f)

            f.write(os.linesep * 2)

            # If there's an existing .py file with the same name, prepend its content
            py_file = self._nvidia_dali_path / self._api / f"{module_path}.py"
            if py_file.exists():
                with py_file.open() as file:
                    # Remove comments from the source code
                    source = tokenize.untokenize(
                        token
                        for token in tokenize.generate_tokens(file.readline)
                        if token.type != tokenize.COMMENT
                    )
                f.write(source)

        return self._module_to_file[module_path]

    def close(self):
        for _, f in self._module_to_file.items():
            f.close()


def gen_all_signatures(nvidia_dali_path: Union[Path, str], api: Api):
    """Generate the signatures for "fn", "ops" or "dynamic" api.

    Parameters
    ----------
    nvidia_dali_path : Path or str
        The path to the wheel pre-packaging to the nvidia/dali directory.
    api : str
        "fn", "ops" or "dynamic"
    """
    nvidia_dali_path = Path(nvidia_dali_path)
    api_path = api if api != "dynamic" else os.path.join("experimental", api)

    with closing(StubFileManager(nvidia_dali_path, api_path)) as stub_manager:
        sig_groups = _group_signatures(api)

        # Python-only and the manually defined ones are reexported from their respective modules
        for schema_name, op in sig_groups["python_only"] + sig_groups["python_wrapper"]:
            _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)

            stub_manager.get(module_nesting).write(
                f"{os.linesep * 2}from {op._impl_module} import"
                f" ({op.__name__} as {op.__name__}){os.linesep * 2}"
            )

        # Re-export pure-Python definitions in DALI dynamic
        # It's possible that some symbols conflict with types used in the annotations.
        if api == "dynamic":
            for module, name in _extract_dynamic_mode_definitions():
                stub_manager.get([]).write(
                    f"{os.linesep * 2}from {module} import" f" ({name} as {name}){os.linesep * 2}"
                )

        # we do not go over sig_groups["hidden_or_internal"] at all as they are supposed to not be
        # directly visible

        # Runtime generated classes use fully specified stubs.
        signature_generators = {
            "fn": _gen_fn_signature,
            "ops": _gen_ops_signature,
            "dynamic": _gen_dynamic_signature,
        }
        for schema_name, op in sig_groups["generated"]:
            _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)
            schema = _b.TryGetSchema(schema_name)

            signature = signature_generators[api](schema, schema_name, op_name)
            stub_manager.get(module_nesting).write(signature)
