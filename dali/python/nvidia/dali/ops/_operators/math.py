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

from nvidia.dali import _conditionals

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.types import (DALIDataType as _DALIDataType, Constant as _Constant, ScalarConstant
                               as _ScalarConstant, _bool_types, _int_like_types, _float_types)


def _is_boolean_like(input):
    """
    Check if scalar constant input provided to arithmetic operator is a boolean constant.

    Parameters
    ----------
    input :
        Input representing scalar constant (not a DataNode or a tensor constant like np.array)
    """
    if type(input) is bool:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _bool_types:
            return True
    return False


def _is_integer_like(input):
    """
    Check if scalar constant input provided to arithmetic operator is an integer constant.

    Boolean and integer types are considered integer-like

    Parameters
    ----------
    input :
        Input representing scalar constant (not a DataNode or a tensor constant like np.array)
    """
    if _is_boolean_like(input):
        return True
    if type(input) is int:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _int_like_types:
            return True
    return False


def _is_real_like(input):
    """
    Check if scalar constant input provided to arithmetic operator is a floating point constant.

    Parameters
    ----------
    input :
        Input representing scalar constant (not a DataNode or a tensor constant like np.array)
    """
    if type(input) is float:
        return True
    if isinstance(input, _ScalarConstant):
        if input.dtype in _float_types:
            return True
    return False


def _to_type_desc(input):
    """
    Generate <type> description required by ArithmeticGenericOp for the usage with scalar constants.
    """
    if type(input) is bool:
        return "bool"
    if type(input) is int:
        return "int32"
    if type(input) is float:
        return "float32"  # TODO(klecki): current DALI limitation
    if isinstance(input, _ScalarConstant):
        dtype_to_desc = {
            _DALIDataType.BOOL: "bool",
            _DALIDataType.INT8: "int8",
            _DALIDataType.INT16: "int16",
            _DALIDataType.INT32: "int32",
            _DALIDataType.INT64: "int64",
            _DALIDataType.UINT8: "uint8",
            _DALIDataType.UINT16: "uint16",
            _DALIDataType.UINT32: "uint32",
            _DALIDataType.UINT64: "uint64",
            _DALIDataType.FLOAT16: "float16",
            _DALIDataType.FLOAT: "float32",
            _DALIDataType.FLOAT64: "float64",
        }
        return dtype_to_desc[input.dtype]

    raise TypeError(
        f"Constant argument to arithmetic operation not supported. "
        f"Got {str(type(input))}, expected "
        f"a constant value of type 'bool', 'int', 'float' or 'nvidia.dali.types.Constant'.")


# Group inputs into categories_idxs, edges of type ``edge_type``,
# integer constants and real constants.
# The categories_idxs is a list that for an input `i` contains a tuple:
# (category of ith input, index of ith input in appropriate category)
def _group_inputs(inputs, edge_type=_DataNode):
    """
    Group inputs into three groups:
       * edges of type ``edge_type`` - those are actual inputs like DataNode,
       * integer constants,
       * real constants.
    Generate `categories_idxs` mapping, that is a list that for an input `i` contains a tuple:
       (category of ith input, index of ith input in appropriate category)
    Parameters
    ----------
    inputs :
        All arguments that were passed to the arithmetic operators
    edge_type :
        What should be considered an input, _DataNode or a TensorList (used for debug and eager
        modes), by default _DataNode

    Returns
    -------
    (`categories_idxs`, input edge category, integer constants category, real constants category)
        Mapping of inputs into the categories and the three possible categories.

    """
    categories_idxs = []
    edges = []
    integers = []
    reals = []
    for input in inputs:
        if not isinstance(input, (edge_type, _ScalarConstant, int, float)):
            input = _Constant(input)
        if isinstance(input, edge_type):
            categories_idxs.append(("edge", len(edges)))
            edges.append(input)
        elif _is_integer_like(input):
            categories_idxs.append(("integer", len(integers)))
            integers.append(input)
        elif _is_real_like(input):
            categories_idxs.append(("real", len(reals)))
            reals.append(input)
        else:
            raise TypeError(f"Argument to arithmetic operation not supported."
                            f"Got {str(type(input))}, expected a return value from other"
                            f"DALI Operator  or a constant value of type 'bool', 'int', "
                            f"'float' or 'nvidia.dali.types.Constant'.")

    if len(integers) == 0:
        integers = None
    if len(reals) == 0:
        reals = None
    return (categories_idxs, edges, integers, reals)


def _generate_input_desc(categories_idx, integers, reals):
    """
    Generate the list of <input> subexpression as specified
    by grammar for ArithmeticGenericOp
    """
    input_desc = ""
    for i, (category, idx) in enumerate(categories_idx):
        if category == "edge":
            input_desc += "&{}".format(idx)
        elif category == "integer":
            input_desc += "${}:{}".format(idx, _to_type_desc(integers[idx]))
        elif category == "real":
            input_desc += "${}:{}".format(idx, _to_type_desc(reals[idx]))
        if i < len(categories_idx) - 1:
            input_desc += " "
    return input_desc


def _arithm_op(name, *inputs):
    """
    Create arguments for ArithmeticGenericOp and call it with supplied inputs.
    Select the `gpu` device if at least one of the inputs is `gpu`, otherwise `cpu`.
    """
    import nvidia.dali.ops  # Allow for late binding of the ArithmeticGenericOp from parent module.
    categories_idxs, edges, integers, reals = _group_inputs(inputs)
    input_desc = _generate_input_desc(categories_idxs, integers, reals)
    expression_desc = "{}({})".format(name, input_desc)
    dev = nvidia.dali.ops._choose_device(edges)
    # Create "instance" of operator
    op = nvidia.dali.ops.ArithmeticGenericOp(device=dev, expression_desc=expression_desc,
                                             integer_constants=integers, real_constants=reals)
    # If we are on gpu, we must mark all inputs as gpu
    if dev == "gpu":
        dev_inputs = list(edge.gpu() for edge in edges)
    else:
        dev_inputs = edges

    # Call it immediately
    result = op(*dev_inputs)
    if _conditionals.conditionals_enabled():
        _conditionals.register_data_nodes(result, dev_inputs)
    return result
