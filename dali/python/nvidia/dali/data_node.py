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

# Delay the evaluation of annotations, so we can use the current class as return type
# for its methods: https://peps.python.org/pep-0563/ (Postponed annotations)
from __future__ import annotations

# pylint: disable=no-member
import sys

from ._utils.hacks import not_iterable
from ._utils import dali_trace as _dali_trace


def _arithm_op(*args, **kwargs):
    import nvidia.dali.ops

    if _dali_trace.is_tracing_enabled():
        definition_frame_end = _dali_trace.get_stack_depth() - 2
    else:
        definition_frame_end = None

    # Fully circular imports don't work. We need to import _arithm_op late and
    # replace this trampoline function.
    setattr(sys.modules[__name__], "_arithm_op", nvidia.dali.ops._arithm_op)
    return nvidia.dali.ops._arithm_op(*args, **kwargs, definition_frame_end=definition_frame_end)


class _NewAxis:
    def __init__(self, name=None):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("Axis name must be a single-character string")
            if len(name) != 1:
                raise ValueError("Axis name must be a single-character string")
        self._name = name

    @property
    def name(self):
        return self._name

    def __call__(self, name=None):
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

    def __str__(self):
        return f'DataNode(name="{self.name}", device="{self.device}, source="{self.source}")'

    __repr__ = __str__

    def gpu(self) -> DataNode:
        return self._to_backend("gpu")

    def cpu(self) -> DataNode:
        self._check_gpu2cpu()
        return self._to_backend("cpu")

    # Note: Regardless of whether we want the cpu or gpu version
    # of a tensor, we keep the source argument the same so that
    # the pipeline can backtrack through the user-defined graph
    def _to_backend(self, backend) -> DataNode:
        if self.device == backend:
            return self

        from nvidia.dali import _conditionals

        if _conditionals.conditionals_enabled():
            # Treat it the same way as regular operator would behave
            [self_split], _ = _conditionals.apply_conditional_split_to_args([self], {})
            transferred_node = DataNode(self_split.name, backend, self_split.source)
            _conditionals.register_data_nodes(transferred_node, [self])
            return transferred_node
        return DataNode(self.name, backend, self.source)

    def __add__(self, other) -> DataNode:
        return _arithm_op("add", self, other)

    def __radd__(self, other) -> DataNode:
        return _arithm_op("add", other, self)

    def __sub__(self, other) -> DataNode:
        return _arithm_op("sub", self, other)

    def __rsub__(self, other) -> DataNode:
        return _arithm_op("sub", other, self)

    def __mul__(self, other) -> DataNode:
        return _arithm_op("mul", self, other)

    def __rmul__(self, other) -> DataNode:
        return _arithm_op("mul", other, self)

    def __pow__(self, other) -> DataNode:
        return _arithm_op("pow", self, other)

    def __rpow__(self, other) -> DataNode:
        return _arithm_op("pow", other, self)

    def __truediv__(self, other) -> DataNode:
        return _arithm_op("fdiv", self, other)

    def __rtruediv__(self, other) -> DataNode:
        return _arithm_op("fdiv", other, self)

    def __floordiv__(self, other) -> DataNode:
        return _arithm_op("div", self, other)

    def __rfloordiv__(self, other) -> DataNode:
        return _arithm_op("div", other, self)

    def __neg__(self) -> DataNode:
        return _arithm_op("minus", self)

    # Short-circuiting the execution, unary + is basically a no-op
    def __pos__(self) -> DataNode:
        return self

    def __eq__(self, other) -> DataNode:
        return _arithm_op("eq", self, other)

    def __ne__(self, other) -> DataNode:
        return _arithm_op("neq", self, other)

    def __lt__(self, other) -> DataNode:
        return _arithm_op("lt", self, other)

    def __le__(self, other) -> DataNode:
        return _arithm_op("leq", self, other)

    def __gt__(self, other) -> DataNode:
        return _arithm_op("gt", self, other)

    def __ge__(self, other) -> DataNode:
        return _arithm_op("geq", self, other)

    def __and__(self, other) -> DataNode:
        return _arithm_op("bitand", self, other)

    def __rand__(self, other) -> DataNode:
        return _arithm_op("bitand", other, self)

    def __or__(self, other) -> DataNode:
        return _arithm_op("bitor", self, other)

    def __ror__(self, other) -> DataNode:
        return _arithm_op("bitor", other, self)

    def __xor__(self, other) -> DataNode:
        return _arithm_op("bitxor", self, other)

    def __rxor__(self, other) -> DataNode:
        return _arithm_op("bitxor", other, self)

    def __bool__(self):
        raise TypeError(
            '"DataNode" was used in conditional context - it might have been used in truth'
            " evaluation for `if` statement, logical expression or cast to a boolean."
            " To use conditional execution via `if` statements you need to specify"
            " `enable_conditionals=True` in `@nvidia.dali.pipeline_def` decorator."
            " You can read more about conditional execution in specific section of the Pipeline"
            " documentation. Bool conversion can be achieved with the `cast` operator."
        )

    def __getitem__(self, val) -> DataNode:
        idxs = []
        new_axes = []
        new_axis_names = []

        # returns True if this index adds a new output dimension
        def process_index(idx, dim):
            if idx is None:
                idxs.append((None, None, None, None))
                return True
            elif isinstance(idx, slice):
                idxs.append((None, idx.start, idx.stop, idx.step))
                return True
            if isinstance(idx, _NewAxis):
                new_axes.append(dim)
                if idx.name is not None:
                    new_axis_names.append(idx.name)
                return True
            if idx is Ellipsis:
                raise NotImplementedError("Ellipsis in indexing is not implemented")
            if isinstance(idx, (float, str)):
                raise TypeError("Invalid type for an index: ", type)
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
                raise ValueError("New axis name must be specified for all axes or none.")
            new_axis_names = "".join(new_axis_names)
        else:
            new_axis_names = None

        slice_args = {}
        for i, (at, lo, hi, step) in enumerate(idxs):
            if at is not None:
                slice_args["at_%i" % i] = at
            if lo is not None:
                slice_args["lo_%i" % i] = lo
            if hi is not None:
                slice_args["hi_%i" % i] = hi
            if step is not None:
                slice_args["step_%i" % i] = step

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

    def shape(self, *, dtype=None, device="cpu"):
        """Returns the run-time shapes of this DataNode as a new DataNode

        Parameters
        ----------
        arg_dtype : DALIDataType, optional
            If specified, the shape will be converted to this data type; defaults to INT64.
        device : str, optional
            The device ("cpu" or "gpu") where the result is returned; defaults to CPU.
        """
        from . import fn

        if device == "cpu":
            self._check_gpu2cpu()
        return fn._shape(self, dtype=dtype, device=device)

    def property(self, key, *, device="cpu"):
        """Returns a metadata property associated with a DataNode

        Parameters
        ----------
        key : str
            The name of the metadata item. Currently supported:
            "source_info"   - the file name or location in the dataset where the data originated
                              (each sample is a 1D uint8 tensor)
            "layout"        - the layout string
                              (each sample is a 1D uint8 tensor)
        device : str, optional
            The device, where the value is returned; defaults to CPU.
        """

        from . import fn

        if device == "cpu":
            self._check_gpu2cpu()

        return fn.get_property(self, key=key, device=device)

    def source_info(self, *, device="cpu"):
        """Returns the "source_info" property. Equivalent to self.meta("source_info")."""
        return self.property("source_info", device=device)

    def _check_gpu2cpu(self):
        """Checks whether using this `DataNode` in a CPU operator is legal.

        The function checks whether it's legal to pass it as an input to a CPU operator.
        If the node is a result of a GPU operator which belongs to a pipeline with non-dynamic
        executor, an error is raised.

        .. note::
        If the defining operator does not yet belong to any pipeline, the error is not raised and
        the check is deferred until `Pipeline.build`.
        """
        if self.device == "gpu" and self.source and self.source.pipeline:
            self.source.pipeline._require_exec_dynamic(
                "This pipeline doesn't support transition from GPU to CPU.\n"
                "GPU->CPU transitions require "
            )


not_iterable(DataNode)


def _check(maybe_node):
    if not isinstance(maybe_node, DataNode):
        raise TypeError(
            f'Expected outputs of type compatible with "DataNode". '
            f'Received output type with name "{type(maybe_node).__name__}" '
            f"that does not match."
        )
