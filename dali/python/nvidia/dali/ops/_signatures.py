# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from inspect import Parameter, Signature
import ast
import os

from pathlib import Path

from contextlib import closing

from typing import Union, Optional
from typing import Sequence, List, Any

from nvidia.dali import backend as _b
from nvidia.dali import types as _types
from nvidia.dali.ops import _registry, _names, _docs
from nvidia.dali import types
from nvidia.dali import ops, fn


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

_enum_mapping = {
    types.DALIDataType: _DALIDataType,
    types.DALIImageType: _DALIImageType,
    types.DALIInterpType: _DALIInterpType
}

_MAX_INPUT_SPELLED_OUT = 10


def _scalar_element_annotation(scalar_dtype):
    # We already have function that converts a scalar constant/literal into the desired type,
    # utilize the fact that they accept integer values and get the actual type.
    conv_fn = _types._known_types[scalar_dtype][1]
    try:
        dummy_val = conv_fn(0)
        t = type(dummy_val)
        if t in _enum_mapping:
            return _enum_mapping[t]
        return t
    # This is tied to TFRecord implementation
    except NotImplementedError:
        return Any
    except TypeError:
        return Any


def _arg_type_annotation(arg_dtype):
    """Convert regular key-word argument type to annotation. Handles Lists and scalars.

    Parameters
    ----------
    arg_dtype : _type_
        _description_
    """
    if arg_dtype in _types._vector_types:
        scalar_dtype = _types._vector_types[arg_dtype]
        scalar_annotation = _scalar_element_annotation(scalar_dtype)
        # DALI allows tuples and lists as a "sequence" parameter
        return Union[Sequence[scalar_annotation], scalar_annotation]
    return _scalar_element_annotation(arg_dtype)


def _get_positional_input_param(schema, idx):
    """Get the Parameter representing positional inputs at `idx`. Automatically mark it as
    optional. The DataNode annotation currently hides the possibility of MIS.

    The double underscore `__` prefix for argument name is an additional way to indicate
    positional only arguments, as per MyPy docs. It is obeyed by the VSCode.

    TODO(klecki): Constant promotions - ArrayLike? Also: Multiple Input Sets.
    """
    # Only first MinNumInputs are mandatory, the rest are optional:
    default = Parameter.empty if idx < schema.MinNumInput() else None
    annotation = _DataNode if idx < schema.MinNumInput() else Optional[_DataNode]
    if schema.HasInputDox():
        return Parameter(f"__{schema.GetInputName(idx)}", kind=Parameter.POSITIONAL_ONLY,
                         default=default, annotation=annotation)
    else:
        return Parameter(f"__input_{idx}", kind=Parameter.POSITIONAL_ONLY, default=default,
                         annotation=annotation)


def _get_positional_input_params(schema):
    """Get the list of positional only inputs to the operator.
    """
    param_list = []
    if not schema.HasInputDox() and schema.MaxNumInput() > _MAX_INPUT_SPELLED_OUT:
        param_list.append(Parameter("input", Parameter.VAR_POSITIONAL, annotation=_DataNode))
    else:
        for i in range(schema.MaxNumInput()):
            param_list.append(_get_positional_input_param(schema, i))
    return param_list


def _get_keyword_params(schema, all_args_optional=False):
    """Get the list of annotated keyword Parameters to the operator.
    """
    param_list = []
    for arg in schema.GetArgumentNames():
        if schema.IsDeprecatedArg(arg):
            # We don't put the deprecated args in the visible API
            continue
        arg_dtype = schema.GetArgumentType(arg)
        kw_annotation = _arg_type_annotation(arg_dtype)
        is_arg_input = schema.IsTensorArgument(arg)

        annotation = Union[_DataNode, kw_annotation] if is_arg_input else kw_annotation
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
            Parameter(name=arg, kind=Parameter.KEYWORD_ONLY, default=default,
                      annotation=annotation))

    # We omit the **kwargs, as we already specified all possible parameters:
    # param_list.append(Parameter("kwargs", Parameter.VAR_KEYWORD))
    # We could add it, but it would behave as catch all.
    return param_list


def _get_implicit_keyword_params(schema, all_args_optional=False):
    """All operators have some additional kwargs, that are not listed in schema, but are
    implicitly used by DALI.
    """
    _ = all_args_optional
    return [
        # TODO(klecki): The default for `device`` is dependant on the input placement (and API).
        Parameter(name="device", kind=Parameter.KEYWORD_ONLY, default=None,
                  annotation=Optional[str]),
        # The name is truly optional
        Parameter(name="name", kind=Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]),
    ]


def _call_signature(schema, include_inputs=True, include_kwargs=True, include_self=False,
                    data_node_return=True, all_args_optional=False) -> Signature:
    """Generate a Signature for given schema.

    Parameters
    ----------
    schema : OpSchema
        Schema for the operator.
    include_inputs : bool, optional
        If positional inputs should be included in the signature, by default True
    include_kwargs : bool, optional
        If keyword arguments should be included in the signature, by default True
    include_self : bool, optional
        Prepend `self` as first positional argument in the signature, by default False
    data_node_return : bool, optional
        If the signature should have a return annotation or return None (for ops class __init__),
        by default True
    all_args_optional : bool, optional
        Make all keyword arguments optional, even if they are not - needed by the ops API, where
        the argument can be specified in either __init__ or __call__, by default False
    """
    param_list = []
    if include_self:
        param_list.append(Parameter("self", kind=Parameter.POSITIONAL_ONLY))

    if include_inputs:
        param_list.extend(_get_positional_input_params(schema))

    if include_kwargs:
        param_list.extend(_get_keyword_params(schema, all_args_optional=all_args_optional))
        param_list.extend(_get_implicit_keyword_params(schema, all_args_optional=all_args_optional))

    if data_node_return:
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
                return_annotation = Sequence[_DataNode]
    else:
        return_annotation = None
    return Signature(param_list, return_annotation=return_annotation)


def inspect_repr_fixups(signature: str) -> str:
    """Replace the weird quirks of printing the repr of signature.
    We use signature object for type safety and additional validation, but the printing rules
    are questionable in some cases. Python type hints advocate the usage of `None` instead of its
    type, but printing a signature would insert NoneType (specifically replacing
    Optional[Union[...]] with Union[..., None] and printing it as Union[..., NoneType]).
    The NoneType doesn't exist as a `types` definition in some Pythons.
    """
    return signature.replace("NoneType", "None")


def _gen_fn_signature(schema, schema_name, fn_name):
    """Write the stub of the fn API function with the docstring, for given operator.
    """
    return inspect_repr_fixups(f"""
def {fn_name}{_call_signature(schema, include_inputs=True, include_kwargs=True)}:
    \"""{_docs._docstring_generator_fn(schema_name)}
    \"""
    ...
""")


def _gen_ops_signature(schema, schema_name, cls_name):
    """Write the stub of the fn API class with the docstring, __init__ and __call__ for given
    operator.
    """
    return inspect_repr_fixups(f"""
class {cls_name}:
    \"""{_docs._docstring_generator(schema_name)}
    \"""
    def __init__{_call_signature(schema, include_inputs=False, include_kwargs=True,
                                 include_self=True, data_node_return=False,
                                 all_args_optional=True)}:
        ...

    def __call__{_call_signature(schema, include_inputs=True, include_kwargs=True,
                                 include_self=True, all_args_optional=True)}:
        \"""{_docs._docstring_generator_call(schema_name)}
        \"""
        ...
""")


# Preamble with license and helper imports for the stub file.
# We need the placeholders for actual Python classes, as the ones that are exported from backend
# don't seem to work with the intellisense.
_HEADER = """
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, Optional
from typing import Sequence, Any

from nvidia.dali.data_node import DataNode

from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

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
        fn or orps
    full_qualified_name : List[str]
        For example ["readers", "File"]
    """
    op = api_module
    for elem in full_qualified_name:
        op = getattr(op, elem, None)
    return op


def _group_signatures(api: str):
    """Divide all operators registered into the "ops" or "fn" api into 4 categories and return them
    as a dictionary:
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
        "generated": []
    }

    api_module = fn if api == "fn" else ops

    for schema_name in sorted(_registry._all_registered_ops()):
        schema = _b.TryGetSchema(schema_name)

        _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)
        op = _get_op(api_module, module_nesting + [op_name])

        if schema is None:
            if op is not None:
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

    def __init__(self, nvidia_dali_path: Path, api: str):
        self._module_to_file = {}
        self._nvidia_dali_path = nvidia_dali_path
        self._api = api
        self._module_tree = _build_module_tree()

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
            f.write(_HEADER)
            full_module_nesting = [""] + module_nesting
            # Find out all the direct submodules and add the imports
            submodules_dict = self._module_tree
            for submodule in full_module_nesting:
                submodules_dict = submodules_dict[submodule]
            direct_submodules = submodules_dict.keys()
            for direct_submodule in direct_submodules:
                f.write(f"from . import {direct_submodule}\n")

            f.write("\n\n")
        return self._module_to_file[module_path]

    def close(self):
        for _, f in self._module_to_file.items():
            f.close()


def gen_all_signatures(nvidia_dali_path, api):
    """Generate the signatures for "fn" or "ops" api.

    Parameters
    ----------
    nvidia_dali_path : Path
        The path to the wheel pre-packaging to the nvidia/dali directory.
    api : str
        "fn" or "ops"
    """
    nvidia_dali_path = Path(nvidia_dali_path)

    with closing(StubFileManager(nvidia_dali_path, api)) as stub_manager:

        sig_groups = _group_signatures(api)

        # Python-only and the manually defined ones are reexported from their respective modules
        for (schema_name, op) in sig_groups["python_only"] + sig_groups["python_wrapper"]:
            _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)

            stub_manager.get(module_nesting).write(f"\n\nfrom {op._impl_module} import"
                                                   f" ({op.__name__} as {op.__name__})\n\n")

        # we do not go over sig_groups["hidden_or_internal"] at all as they are supposed to not be
        # directly visible

        # Runtime generated classes use fully specified stubs.
        for (schema_name, op) in sig_groups["generated"]:
            _, module_nesting, op_name = _names._process_op_name(schema_name, api=api)
            schema = _b.TryGetSchema(schema_name)

            if api == "fn":
                stub_manager.get(module_nesting).write(
                    _gen_fn_signature(schema, schema_name, op_name))
            else:
                stub_manager.get(module_nesting).write(
                    _gen_ops_signature(schema, schema_name, op_name))
