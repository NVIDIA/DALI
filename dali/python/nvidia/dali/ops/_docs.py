# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import backend as _b
from nvidia.dali.types import _default_converter, _type_name_convert_to_string

from nvidia.dali.ops import _registry, _names


def _numpydoc_formatter(name, type, doc, optional=False):
    """
    Format the documentation for single argument, `name`, `type` and `doc` are expected to be
    strings.

    The formatting is:
    <name> : <type>[, optional]
        <doc>
    """
    indent = "\n" + " " * 4
    if optional:
        type += ", optional"
    return "{} : {}{}{}".format(name, type, indent, doc.replace("\n", indent))


def _get_inputs_doc(schema):
    """
    Generate numpydoc-formatted docstring section for operator inputs (positional arguments)
    based on the schema.

    The inputs are represented in `Args` section using `_numpydoc_formatter`.

    If schema provides names and docstrings for inputs, they are used, otherwise placeholder
    text is used indicating the supported number of inputs.

    Note: The type of input is indicated as TensorList with supported layouts listed.

    schema : OpSchema
       Schema of the operator to be documented
    """
    # Inputs section
    if schema.MaxNumInput() == 0:
        return ""
    ret = """
Args
----
"""
    if schema.HasInputDox():
        for i in range(schema.MaxNumInput()):
            optional = i >= schema.MinNumInput()
            input_type_str = schema.GetInputType(i) + _supported_layouts_str(
                schema.GetSupportedLayouts(i)
            )
            dox = schema.GetInputDox(i)
            input_name = _names._get_input_name(schema, i)
            ret += _numpydoc_formatter(input_name, input_type_str, dox, optional) + "\n"
    else:
        for i in range(schema.MinNumInput()):
            input_type_str = "TensorList" + _supported_layouts_str(schema.GetSupportedLayouts(i))
            dox = "Input to the operator."
            input_name = _names._get_input_name(schema, i)
            ret += _numpydoc_formatter(input_name, input_type_str, dox, False) + "\n"

        extra_opt_args = schema.MaxNumInput() - schema.MinNumInput()
        if extra_opt_args == 1:
            i = schema.MinNumInput()
            input_type_str = "TensorList" + _supported_layouts_str(schema.GetSupportedLayouts(i))
            dox = "Input to the operator."
            input_name = _names._get_input_name(schema, i)
            ret += _numpydoc_formatter(input_name, input_type_str, dox, True) + "\n"
        elif extra_opt_args > 1:
            input_type_str = "TensorList"
            generic_name = _names._get_generic_input_name(False)
            input_name = f"{generic_name}[{schema.MinNumInput()}..{schema.MaxNumInput()-1}]"
            dox = f"This function accepts up to {extra_opt_args} optional positional inputs"
            ret += _numpydoc_formatter(input_name, input_type_str, dox, True) + "\n"

    ret += "\n"
    return ret


def _get_kwargs(schema):
    """
    Get the numpydoc-formatted docstring section for keywords arguments.


    schema : OpSchema
       Schema of the operator to be documented
    """
    ret = ""
    for arg in schema.GetArgumentNames():
        skip_full_doc = False
        type_name = ""
        dtype = None
        doc = ""
        deprecation_warning = None
        if schema.IsDeprecatedArg(arg):
            meta = schema.DeprecatedArgInfo(arg)
            msg = meta["msg"]
            assert msg is not None
            deprecation_warning = ".. warning::\n\n    " + msg.replace("\n", "\n    ")
            renamed_arg = meta["renamed_to"]
            # Renamed and removed arguments won't show full documentation (only warning box)
            skip_full_doc = renamed_arg or meta["removed"]
            # Renamed aliases are not fully registered to the schema, that's why we query for the
            # info on the renamed_arg name.
            if renamed_arg:
                dtype = schema.GetArgumentType(renamed_arg)
                type_name = _type_name_convert_to_string(
                    dtype, allow_tensors=schema.IsTensorArgument(renamed_arg)
                )
        # Try to get dtype only if not set already
        # (renamed args go through a different path, see above)
        if not dtype:
            dtype = schema.GetArgumentType(arg)
            type_name = _type_name_convert_to_string(
                dtype, allow_tensors=schema.IsTensorArgument(arg)
            )
        # Add argument documentation if necessary
        if not skip_full_doc:
            if schema.IsArgumentOptional(arg):
                type_name += ", optional"
                if schema.HasArgumentDefaultValue(arg):
                    default_value_string = schema.GetArgumentDefaultValueString(arg)
                    default_value = ast.literal_eval(default_value_string)
                    type_name += ", default = `{}`".format(_default_converter(dtype, default_value))
            doc += schema.GetArgumentDox(arg).rstrip("\n")
            if schema.ArgSupportsPerFrameInput(arg):
                doc += "\n\nSupports :func:`per-frame<nvidia.dali.fn.per_frame>` inputs."
            if deprecation_warning:
                doc += "\n\n" + deprecation_warning
        elif deprecation_warning:
            doc += deprecation_warning
        ret += _numpydoc_formatter(arg, type_name, doc)
        ret += "\n"
    return ret


def _docstring_generator_main(schema_name, api):
    """
    Generate docstring for the class obtaining it from schema based on cls.__name__
    or the schema name as a str.
    This lists all the Keyword args that can be used when creating operator
    """
    schema = _b.GetSchema(schema_name)
    ret = "\n"

    if schema.IsDeprecated():
        ret += ".. warning::\n\n   This operator is now deprecated."
        replacement = schema.DeprecatedInFavorOf()
        if replacement:
            use_instead = _names._op_name(replacement, api)
            ret += " Use :meth:`" + use_instead + "` instead."
        explanation = schema.DeprecationMessage()
        if explanation:
            indent = "\n" + " " * 3
            ret += indent
            ret += indent
            explanation = explanation.replace("\n", indent)
            ret += explanation
        ret += "\n\n"

    ret += schema.Dox()
    ret += "\n"

    if schema.IsDocPartiallyHidden():
        return ret

    supported_statements = []
    if schema.IsSequenceOperator():
        supported_statements.append("expects sequence inputs")
    elif schema.AllowsSequences():
        supported_statements.append("allows sequence inputs")

    if schema.SupportsVolumetric():
        supported_statements.append("supports volumetric data")

    if len(supported_statements) > 0:
        ret += "\nThis operator "
        ret += supported_statements[0]
        if len(supported_statements) > 1:
            ret += " and " + supported_statements[1]
        ret += ".\n"

    if schema.IsNoPrune():
        ret += "\nThis operator will **not** be optimized out of the graph.\n"

    op_dev = []
    if schema_name in _registry.cpu_ops():
        op_dev.append("'cpu'")
    if schema_name in _registry.gpu_ops():
        op_dev.append("'gpu'")
    if schema_name in _registry.mixed_ops():
        op_dev.append("'mixed'")
    ret += """
Supported backends
"""
    for dev in op_dev:
        ret += " * " + dev + "\n"
    ret += "\n"
    return ret


def _docstring_generator(schema_name):
    schema = _b.GetSchema(schema_name)
    ret = _docstring_generator_main(schema_name, "ops")
    if schema.IsDocPartiallyHidden():
        return ret
    ret += """
Keyword args
------------
"""
    ret += _get_kwargs(schema)
    return ret


def _supported_layouts_str(supported_layouts):
    if len(supported_layouts) == 0:
        return ""
    return " (" + ", ".join(["'" + str(layout) + "'" for layout in supported_layouts]) + ")"


def _docstring_prefix_from_inputs(op_name):
    """
    Generate start of the docstring for `__call__` of Operator `op_name`
    assuming the docstrings were provided for all inputs separately

    Returns list of `Args` in appropriate section
    """
    schema = _b.GetSchema(op_name)
    # __call__ docstring
    ret = "\nOperator call to be used in graph definition.\n"
    # Args section
    ret += _get_inputs_doc(schema)
    return ret


def _docstring_prefix_auto(op_name):
    """
    Generate start of the docstring for `__call__` of Operator `op_name`
    with default values. Assumes there will be 0 or 1 inputs
    """
    schema = _b.GetSchema(op_name)
    if schema.MaxNumInput() == 0:
        return """
Operator call to be used in graph definition. This operator doesn't have any inputs.
"""
    elif schema.MaxNumInput() == 1:
        input_name = _names._get_input_name(schema, 0)
        ret = """
Operator call to be used in graph definition.

Args
----
"""
        dox = "Input to the operator.\n"
        fmt = "TensorList" + _supported_layouts_str(schema.GetSupportedLayouts(0))
        ret += _numpydoc_formatter(input_name, fmt, dox, optional=False)
        return ret
    return ""


def _docstring_generator_call(op_name):
    """
    Generate full docstring for `__call__` of Operator `op_name`.
    """
    schema = _b.GetSchema(op_name)
    if schema.IsDocPartiallyHidden():
        return ""
    if schema.HasCallDox():
        ret = schema.GetCallDox()
    elif schema.HasInputDox():
        ret = _docstring_prefix_from_inputs(op_name)
    elif schema.CanUseAutoInputDox():
        ret = _docstring_prefix_auto(op_name)
    else:
        op_full_name, _, _ = _names._process_op_name(op_name)
        ret = "See :meth:`nvidia.dali.ops." + op_full_name + "` class for complete information.\n"
    if schema.AppendKwargsSection():
        # Kwargs section
        tensor_kwargs = _get_kwargs(schema)
        if tensor_kwargs:
            ret += """
Keyword Args
------------
"""
            ret += tensor_kwargs
    return ret


def _docstring_generator_fn(schema_name):
    schema = _b.GetSchema(schema_name)
    ret = _docstring_generator_main(schema_name, "fn")
    if schema.IsDocPartiallyHidden():
        return ret
    ret += _get_inputs_doc(schema)
    ret += """
Keyword args
------------
"""
    ret += _get_kwargs(schema)
    return ret
