# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from . import _utils
from ._utils import hacks

def _arithm_op(*args, **kwargs):
    import nvidia.dali.ops
    # Fully circular imports don't work. We need to import _arithm_op late and
    # replace this trampoline function.
    setattr(sys.modules[__name__], "_arithm_op", nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs)

class _NewAxis:
    def __init__(self, name = None):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("Axis name must be a single-character string")
            if len(name) != 1:
                raise ValueError("Axis name must be a single-character string")
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, name = None):
        return _NewAxis(name)

newaxis = _NewAxis()

class DataNode(object):
    """This class is a symbolic representation of a TensorList and is used at graph definition
    stage. It does not carry actual data, but is used to define the connections between operators
    and to specify the pipeline outputs. See documentation for :class:`Pipeline` for details.

    ``DataNode`` objects can be passed to DALI operators as inputs (and some of the named keyword
    arguments) but they also provide arithmetic operations which implicitly create appropriate
    operators that perform the expressions.
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

    def __getitem__(self, val):
        idxs = []
        new_axes = []
        new_axis_names = []

        # returns True if this index adds a new output dimension
        def process_index(idx, dim):
            if idx is None:
                idxs.append((None, None, None, None))
                return True
            elif isinstance(idx, slice):
                if idx.step is not None and idx.step != 1:
                    raise NotImplementedError("Slicing with non-unit step is not implemented.")
                idxs.append((None, idx.start, idx.stop, None))
                return True
            elif isinstance(idx, _NewAxis):
                new_axes.append(dim)
                if idx.name is not None:
                    new_axis_names.append(idx.name)
                return True
            elif idx is Ellipsis:
                raise NotImplementedError("Ellipsis in indexing is not implemented")
            elif isinstance(idx, (float, str)):
                raise TypeError("Invalid type for an index: ", type)
            else:
                idxs.append((idx, None, None, None))
                return False

        if not isinstance(val, tuple):
            val = (val,)
        d = 0
        for v in val:
            if process_index(v, d):
                d += 1

        if len(new_axis_names) != 0:
            if len(new_axis_names) != len(new_axes):
                raise ValueError("New axis name must be specified for all axes or none.");
            new_axis_names = "".join(new_axis_names)
        else:
            new_axis_names = None

        slice_args = {}
        for i, (at, lo, hi, step) in enumerate(idxs):
            if at   is not None: slice_args["at_%i"%i] = at
            if lo   is not None: slice_args["lo_%i"%i] = lo
            if hi   is not None: slice_args["hi_%i"%i] = hi
            if step is not None: slice_args["step_%i"%i] = step

        import nvidia.dali.fn
        if len(slice_args) == 0:
            # No true slicing arguments - only full range : and dali.newaxis.
            # We need to ensure there are enough dimensions in the input for the number of
            # full-range axes.
            # If the last index is a newaxis, then ExpandDims will make sure that it makes sense.
            # Otherwise we need to add an additional check.
            if len(new_axes) > 0 and isinstance(val[-1], _NewAxis):
                sliced = self  # no check needed, ExpandDims will do the trick
            else:
                sliced = nvidia.dali.fn.subscript_dim_check(self, num_subscripts=len(idxs))
        else:
            sliced = nvidia.dali.fn.tensor_subscript(self, **slice_args, num_subscripts=len(idxs))
        if len(new_axes) == 0:
            return sliced
        else:
            return nvidia.dali.fn.expand_dims(sliced, axes=new_axes, new_axis_names=new_axis_names)

_utils.hacks.not_iterable(DataNode)

def _check(maybe_node):
    if not isinstance(maybe_node, DataNode):
        raise TypeError(("Expected outputs of type compatible with \"DataNode\"."
                " Received output type with name \"{}\" that does not match.")
                .format(type(maybe_node).__name__))
