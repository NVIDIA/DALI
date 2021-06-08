# Copyright (c) 2017-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pylint: disable=no-member

import sys

def _arithm_op(*args, **kwargs):
    import nvidia.dali.ops
    # Fully circular imports don't work. We need to import _arithm_op late and
    # replace this trampoline function.
    setattr(sys.modules[__name__], "_arithm_op", nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs)


class DataNode(object):
    """This class is a symbolic representation of a TensorList and is used at graph definition
    stage. It does not carry actual data, but is used to define the connections between operators
    and to specify the pipeline outputs. See documentation for :class:`Pipeline` for details.

    `DataNode` objects can be passed to DALI operators as inputs (and some of the named arguments)
    but they also provide arithmetic operations which implicitly create appropriate operators that
    perform the expressions.
    """
    def __init__(self, name, device="cpu", source=None):
        self.name = name
        self.device = device
        self.source = source

    # Note: Regardless of whether we want the cpu or gpu version
    # of a tensor, we keep the source argument the same so that
    # the pipeline can backtrack through the user-defined graph
    def gpu(self):
        return DataNode(self.name, "gpu", self.source)

    def __add__(self, other):
        return _arithm_op("add", self, other)
    def __radd__(self, other):
        return _arithm_op("add", other, self)

    def __sub__(self, other):
        return _arithm_op("sub", self, other)
    def __rsub__(self, other):
        return _arithm_op("sub", other, self)

    def __mul__(self, other):
        return _arithm_op("mul", self, other)
    def __rmul__(self, other):
        return _arithm_op("mul", other, self)

    def __pow__(self, other):
        return _arithm_op("pow", self, other)
    def __rpow__(self, other):
        return _arithm_op("pow", other, self)

    def __truediv__(self, other):
        return _arithm_op("fdiv", self, other)
    def __rtruediv__(self, other):
        return _arithm_op("fdiv", other, self)

    def __floordiv__(self, other):
        return _arithm_op("div", self, other)
    def __rfloordiv__(self, other):
        return _arithm_op("div", other, self)

    def __neg__(self):
        return _arithm_op("minus", self)

    # Short-circuitng the execution, unary + is basically a no-op
    def __pos__(self):
        return self

    def __eq__(self, other):
        return _arithm_op("eq", self, other)

    def __ne__(self, other):
        return _arithm_op("neq", self, other)

    def __lt__(self, other):
        return _arithm_op("lt", self, other)

    def __le__(self, other):
        return _arithm_op("leq", self, other)

    def __gt__(self, other):
        return _arithm_op("gt", self, other)

    def __ge__(self, other):
        return _arithm_op("geq", self, other)

    def __and__(self, other):
        return _arithm_op("bitand", self, other)
    def __rand__(self, other):
        return _arithm_op("bitand", other, self)

    def __or__(self, other):
        return _arithm_op("bitor", self, other)
    def __ror__(self, other):
        return _arithm_op("bitor", other, self)

    def __xor__(self, other):
        return _arithm_op("bitxor", self, other)
    def __rxor__(self, other):
        return _arithm_op("bitxor", other, self)

    def __bool__(self):
        raise TypeError(("\"DataNode\" is a symbolic representation of TensorList used for defining"
                " graph of operations for DALI Pipeline. It should not be used for truth evaluation"
                " in regular Python context. Bool conversion in Pipeline can be achieved"
                " with \"Cast\" operator. To see what operations are allowed on DataNodes to"
                " represent computations in DALI Pipeline see the \"Mathematical Expressions\""
                " section of DALI documentation."))


def _check(maybe_node):
    if not isinstance(maybe_node, DataNode):
        raise TypeError(("Expected outputs of type compatible with \"DataNode\"."
                " Received output type with name \"{}\" that does not match.")
                .format(type(maybe_node).__name__))
