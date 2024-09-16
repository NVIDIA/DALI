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
"""This module contains the implementation of DALI if statement.

It initializes AutoGraph with the DaliOperatorOverload that provides the overload for the if_stmt
and adjust the filtered modules so DALI code is not converted.

The if_stmt provides access to both branches as callables and the set_state/get_state functions
that allows to capture and adjust all symbols modified within those branches. This allows to
checkpoint the state and visit the code of both branches.

if_stmt highlights which state variables are considered the outputs of the if/else pair - we can
use the state captured after visiting if and else branches and produce fn._conditional.merge
nodes for all of them.

When visiting the if/else scopes, we are tracking tha path that we took and the predicates that
were used via the _ConditionStack. As it is not easy to detect which state variables would be
consumed as inputs to DALI operators, we inject additional code to the operator function.
Every time a DataNode is consumed, we look up in which scope it was produced and travel the
path from that point to the current scope in the _ConditionStack, applying necessary splits.
All the return values are registered to the current scope for further lookups.
"""

from nvidia.dali import _autograph
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali import fn

from nvidia.dali._autograph.utils import ag_logging as logging
from nvidia.dali._autograph.operators import variables

from contextlib import contextmanager

from enum import Enum

import tree


def _data_node_repr(data_node):
    return f"DataNode(name={data_node.name}, device={data_node.device}, source={data_node.source})"


def _map_structure(func, *structures, **kwargs):
    """Custom wrapper over tree.map_structure that filters it out from the user-visible stack trace
    for error reporting purposes.
    """
    with _autograph.CustomModuleFilter(tree):
        return tree.map_structure(func, *structures, **kwargs)


class _Branch(Enum):
    TrueBranch = 0
    FalseBranch = 1
    Undefined = 2


class _StackEntry:
    """Information about 1 nesting level of if/else statement.

    Keeps the current branch (if we entered if/else branch) and the data nodes that were
    produced in their scopes. Keeps the mapping of DataNodes produced in higher scopes that
    were already split for use in this scope.
    """

    def __init__(self, predicate):
        self.predicate = predicate
        self.branch = _Branch.Undefined
        self.splits = {}
        self.produced_true = set()
        self.produced_false = set()
        # The produced_special handles the case of producing something visible on the same nesting
        # level, but not in one of the branches and is used by merge code.
        self.produced_special = set()

    @property
    def produced(self):
        """
        Access the set of hashes of DataNodes produced in the scope of currently selected branch.
        """
        if self.branch == _Branch.TrueBranch:
            return self.produced_true
        elif self.branch == _Branch.FalseBranch:
            return self.produced_false
        else:
            return self.produced_special | self.produced_true | self.produced_false

    @produced.setter
    def produced(self, value):
        """
        Access the set of hashes of DataNodes produced in the scope of currently selected branch
        """
        if self.branch == _Branch.TrueBranch:
            self.produced_true = value
        elif self.branch == _Branch.FalseBranch:
            self.produced_false = value
        else:
            self.produced_special = value

    def add_produced(self, data_node):
        """Add the DataNode or DataNodes to produced in the scope of currently selected branch."""
        if isinstance(data_node, _DataNode):
            self.produced |= {_data_node_repr(data_node)}
        elif isinstance(data_node, list):
            if not data_node:
                return
            if isinstance(data_node[0], _DataNode):
                self.produced |= set(_data_node_repr(dn) for dn in data_node)
            elif isinstance(data_node[0], list):
                flat_list = [item for sublist in data_node for item in sublist]
                self.add_produced(flat_list)
        else:
            raise ValueError(
                f"Unexpected operator result to register: {data_node}. Expected up to"
                " two-level nesting of DataNode."
            )

    def add_split(self, source_data_node, producer_node, true_node, false_node):
        """Register the outputs of split node that were produced from the source_data_node
        (or its descendant on this scope, the shortcut node).

        Parameters
        ----------
        source_data_node : DataNode
            Original source node that was looked up, record for faster consecutive lookups
        producer_node : DataNode
            The closest node on the path from source_data_node to this split
        true_node : DataNode
            True branch split
        false_node : DataNode
            False branch split
        """
        self.splits[_data_node_repr(source_data_node)] = (true_node, false_node)
        # Record the direct preceding node as the producer:
        self.splits[_data_node_repr(producer_node)] = (true_node, false_node)
        self.produced_true |= {_data_node_repr(true_node)}
        self.produced_false |= {_data_node_repr(false_node)}

    def __str__(self):
        return (
            f"StackEntry: pred={self.predicate}, branch={self.branch}, splits={self.splits},"
            f" produced={self.produced}"
        )

    def has(self, data_node):
        """
        Check if this DataNode was either produced in this scope or already split for this scope.
        """
        if _data_node_repr(data_node) in self.produced:
            return True
        elif _data_node_repr(data_node) in self.splits:
            return True
        else:
            return False

    def get(self, data_node):
        """Return the `data_node` if it was produced in this scope, or the appropriate split node
        that was created for accessing the `data_node` in this scope.
        """
        assert self.has(data_node)
        if _data_node_repr(data_node) in self.produced:
            return data_node
        else:
            assert self.branch in {_Branch.TrueBranch, _Branch.FalseBranch}
            return self.splits[_data_node_repr(data_node)][self.branch.value]


class _ConditionStack:
    """Tracks the current if/else scope with the path that we took. Captures the used and produced
    data nodes, applying the necessary splits based on the scope level where they were produced
    and where they are used.
    """

    def __init__(self):
        self._stack = [_StackEntry(None)]
        self._is_registration_allowed = True

    def push_predicate(self, predicate):
        """Add next level of if/else scope that is predicated with the `predicate`.
        The user might have provided a predicate from a scope of higher level, which means
        that `predicate` might be subject to additional slicing. Apply that slicing and return
        the actual predicate that will be used for slicing when entering this scope.

        The situation will happen for example in a case like this, where both predicates are
        produced in global scope:

        pred_0 = ...
        pred_1 = ...

        if pred_0:  # push_pred(pred_0) -> returns pred_0
            if pred_1:  # push_pred(pred_1) ->
                        # -> returns fn._conditional.slice(pred_1, predicate=pred_0)

        Parameters
        ----------
        predicate : DataNode
            Predicate guarding this scope.

        Returns
        -------
        DataNode
            Actual predicate after applying necessary slices to use it in this scope.
        """
        new_pred = self.preprocess_input(predicate)
        new_entry = _StackEntry(new_pred)
        self._stack.append(new_entry)
        return new_pred

    def top(self):
        """Get the top scope in the stack"""
        return self._stack[-1]

    def pop(self):
        """Remove the top scope from the stack"""
        result = self._stack.pop()
        return result

    def stack_depth(self):
        """Get the depth of the stack. Note, that by default there is at least one element
        - the global scope."""
        return len(self._stack)

    def _find_closest(self, data_node):
        """Find the closest scope level in the stack where we can access this node as produced
        (or the split of this node closest to us).
        """
        for level in range(self.stack_depth() - 1, -1, -1):
            if self._stack[level].has(data_node):
                return level

        raise ValueError(f"{data_node} was not produced within this trace.")

    def _realize_split(self, data_node, stack_level):
        """The data_node was produced (or last accessed as via split) in scope earlier than the
        current one, traverse the scopes between that level and current one, and insert split nodes.

        Parameters
        ----------
        data_node : DataNode
            The data node that we want to use in the current scope.
        stack_level : int
            Stack level where the data_node was last "seen".

        Returns
        -------
        DataNode
            New node that can be used in current branch and scope.
        """
        assert 0 <= stack_level and stack_level < self.stack_depth() - 1
        produced_data_node = self._stack[stack_level].get(data_node)
        bottom = self._stack[: stack_level + 1]
        top = self._stack[stack_level + 1 :]
        self._stack = bottom
        while top:
            current_entry = top.pop(0)
            predicate = current_entry.predicate

            # Do not automatically register the outputs in the current scope, we track them below
            # in their respective branches.
            logging.log(
                9,
                (
                    f"{self._indent()}[IF] Inserting split"
                    f" at {self.stack_depth() -1}:"
                    f" split({produced_data_node}, predicate={predicate}."
                ),
            )
            self._is_registration_allowed = False
            true_node, false_node = fn._conditional.split(
                produced_data_node, predicate=predicate, _if_stmt=True
            )
            self._is_registration_allowed = True

            # Record the result of splitting the `data_node` that we are trying to look up
            # (short-cut for consecutive lookups)
            current_entry.add_split(data_node, produced_data_node, true_node, false_node)
            if current_entry.branch == _Branch.TrueBranch:
                produced_data_node = true_node
            else:
                produced_data_node = false_node
            self._stack.append(current_entry)
        return produced_data_node

    def preprocess_input(self, data_node):
        """Process the DataNode that is an input to an operator call. Detect if the DataNode was
        produced on the same nesting level. If not, split accordingly to the stack of the previous
        conditions. Caches the previously processed DataNodes to not do repeated splitting.
        """
        stack_level = self._find_closest(data_node)
        logging.log(
            8,
            (
                f"{self._indent()}[IF/Input] {data_node} accessed at level"
                f" {self.stack_depth() - 1} found at {stack_level}."
            ),
        )
        # We already have it cached or produced in this scope.
        if stack_level == self.stack_depth() - 1:
            return self.top().get(data_node)
        # otherwise, we need to fill in the splits.
        return self._realize_split(data_node, stack_level)

    def register_data_nodes(self, data_nodes, global_scope=False):
        """Register the data nodes as produced in current scope, otherwise if `global_scope` is True
        put them in the outermost scope.
        """
        if not self._is_registration_allowed:
            return
        logging.log(8, (f"{self._indent()}[IF/Register] {data_nodes} at {self.stack_depth() -1}"))
        scope = self._stack[0] if global_scope else self.top()
        _map_structure(lambda node: scope.add_produced(node), data_nodes)

    def track_true_branch(self):
        """Mark `if` (true) branch as current scope."""
        self.top().branch = _Branch.TrueBranch

    def track_false_branch(self):
        """Mark `else` (false) branch as current scope."""
        self.top().branch = _Branch.FalseBranch

    def no_branch(self):
        """Mark no branch being tracked, the scope "level" stays related to the same if/else
        statement."""
        self.top().branch = _Branch.Undefined

    def track_merge(self, split_predicate):
        """Enter the merge section of the if/else statement. It adds the corresponding
        split_predicate to the nodes visible as produced in the current scope, so all data nodes
        are directly accessible in this scope when looked up by the merge operator.
        We don't care about removing it as it's the last thing happening in that statement.
        """
        self.no_branch()
        self.top().add_produced(split_predicate)

    def scope_batch_size_tracker(self):
        """Return the DataNode that can be used as a reference batch size in this scope.
        None is returned if we are in the top level scope.
        """
        if self.stack_depth() == 1:
            return None
        if self.top().branch in {_Branch.TrueBranch, _Branch.FalseBranch}:
            # In worst case we will introduce a split on the predicate itself, but we know,
            # we can consistently do it, and it will happen only for the first operator call,
            # for the following ones in this scope it will be cached.
            return self.preprocess_input(self.top().predicate)
        else:
            # If we are in the merge stage, just use the size of the predicate
            return self.top().predicate

    def _indent(self):
        """Helper for indenting the log messages to resemble visited scopes"""
        return "  " * (self.stack_depth() - 1)


@contextmanager
def _cond_manager(predicate):
    actual_predicate = this_condition_stack().push_predicate(predicate)
    logging.log(
        7,
        (
            f"{this_condition_stack()._indent()}[IF]: {predicate}"
            f" at {this_condition_stack().stack_depth() - 1}"
        ),
    )
    # Return it so we can use it in merge
    yield actual_predicate
    this_condition_stack().pop()


@contextmanager
def _cond_true():
    this_condition_stack().track_true_branch()
    logging.log(
        7,
        (
            f"{this_condition_stack()._indent()}[IF]: `if` branch"
            f" at {this_condition_stack().stack_depth() - 1}"
        ),
    )
    yield
    this_condition_stack().no_branch()


@contextmanager
def _cond_false():
    this_condition_stack().track_false_branch()
    logging.log(
        7,
        (
            f"{this_condition_stack()._indent()}[IF]: `else` branch"
            f" at {this_condition_stack().stack_depth() - 1}"
        ),
    )
    yield
    this_condition_stack().no_branch()


@contextmanager
def _cond_merge(split_predicate):
    this_condition_stack().track_merge(split_predicate)
    yield
    this_condition_stack().no_branch()


def conditionals_enabled():
    """Check (within a Pipeline context) if the conditionals are enabled."""
    from nvidia.dali._debug_mode import _PipelineDebug

    current_pipeline = _PipelineDebug.current()
    enabled = getattr(current_pipeline, "_conditionals_enabled", False)
    return enabled


def this_condition_stack():
    """Return the condition stack of current Pipeline"""
    from nvidia.dali._debug_mode import _PipelineDebug

    current_pipeline = _PipelineDebug.current()
    if current_pipeline._condition_stack is None:
        raise ValueError(
            "Cannot access current condition stack when conditionals"
            " were not enabled for a given pipeline."
        )
    return current_pipeline._condition_stack


def register_data_nodes(data_node, inputs=[], args={}):
    """Register the outputs of the operator as produced in the scope of the current conditional
    branch. Pass the list of inputs and dictionary of arguments to automatically detect if any
    DataNode was passed to that operator, indicating that it has proper inputs or argument inputs
    and can infer the batch size. Otherwise the outputs are registered in global scope, assuming
    that they use current batch size.

    Parameters
    ----------
    data_node : DataNode or a list/tuple of DataNode
        The output of the operator to be registered.
    inputs : List of DataNode
        Optional list of inputs of the operator whose outputs we are registering.
    args : Dict of DataNode
        Optional dictionary containing the arguments of the operator whose outputs we are
        registering.
    """

    any_positional_input = any(isinstance(input, _DataNode) for input in inputs)
    any_arg_input = any(isinstance(arg, _DataNode) for arg_name, arg in args.items())
    any_input = any_positional_input or any_arg_input
    # TODO(klecki): In theory we have two approaches for inputless operators. Here we insert their
    # outputs to top level and let the automatic splitting handle the situation. Otherwise we could
    # pass the scope information and batch_size within that scope to all operators that are invoked
    # within that scope.
    this_condition_stack().register_data_nodes(data_node, global_scope=not any_input)


def inject_implicit_scope_argument(schema, kwargs):
    """
    Adds hidden _scope argument to the inputless operators whose outputs for
    any given sample depend on the actual batch size, e.g. fn.batch_permutation.
    """
    # TODO(ktokarski) Consider optimizing the case - the implicit `_scope` argument is not
    # needed if operator can accept any other positional/tensor argument and any such
    # arg was specified. For now, the ops that use the _scope arg (ImplicitScopeAttr schema)
    # do not have any other tensor inputs.
    if schema.HasArgument("_scope"):
        conditional_scope = this_condition_stack()
        scope_masked_batch = conditional_scope.scope_batch_size_tracker()
        kwargs["_scope"] = scope_masked_batch


def apply_conditional_split(input):
    """Preprocess the DataNode to obtain correctly split batch for the current if scope."""
    return this_condition_stack().preprocess_input(input)


def apply_conditional_split_to_branch_outputs(branch_outputs, promote_constants=True):
    """Apply splitting to the branch outputs. This may be necessary for DataNodes that are
    branch outputs but were not touched in that branch (for example that branch is no-op).

    Parameters
    ----------
    branch_outputs : tuple of DataNode
        Outputs of the branch
    promote_constants : bool, optional
        Whether to promote constants to cpu-based Constant op, by default True

    Returns
    -------
    tuple of DataNode
    """
    from nvidia.dali.types import Constant

    def apply_split(atom):
        if isinstance(atom, _DataNode):
            return apply_conditional_split(atom)
        elif promote_constants:
            # We assume that any return from the branch must be merged, so constants are promoted
            # to batches using constant op, and thus can be used in merge.
            constant_node = Constant(atom, device="cpu")
            register_data_nodes(constant_node)
            return apply_conditional_split(constant_node)
        return atom

    return _map_structure(apply_split, branch_outputs)


def apply_conditional_split_to_args(inputs, kwargs):
    """Preprocess the inputs and kwargs of the operator to obtain correctly split inputs for the
    current if scope."""
    inputs = apply_conditional_split_to_branch_outputs(inputs, False)
    for key, arg in kwargs.items():
        if isinstance(arg, _DataNode):
            kwargs[key] = apply_conditional_split(arg)
    return inputs, kwargs


def _verify_branch_outputs(outputs, symbol_names, branch_name):
    """Verifies variables output by a conditional branch for consistency."""
    common_explanation = (
        "Encountered inconsistent outputs out of the `if/else` control flow statement."
        " Variables need to be initialized in every code path (both `if` branches)."
    )
    for name, output in zip(symbol_names, outputs):
        if isinstance(output, variables.Undefined):
            raise RuntimeError(
                f"{common_explanation} Variable '{name}' must also be initialized"
                f" in the `{branch_name}` branch."
            )
        if isinstance(output, variables.UndefinedReturnValue):
            raise RuntimeError(
                f"{common_explanation} The `{branch_name}` branch must also have"
                f" a return statement."
            )


def _validate_logical(value, expression_name, expression_side):
    v = fn._conditional.validate_logical(
        value, expression_name=expression_name, expression_side=expression_side
    )
    if v.device != "cpu":
        raise RuntimeError(
            f"Logical expression `{value}` is restricted to scalar (0-d tensors)"
            f" inputs of `bool` type, that are placed on CPU."
            f" Got a GPU input as the {expression_side} argument in logical expression."
        )
    return v


class DaliOperatorOverload(_autograph.OperatorBase):
    def detect_overload_ld(self, v):
        return isinstance(v, _DataNode)

    def ld(self, v):
        branch_v = apply_conditional_split(v)
        return branch_v

    def detect_overload_if_stmt(self, cond):
        return isinstance(cond, _DataNode)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        # Initial checkpoint before if
        init_state = get_state()
        with _cond_manager(cond) as split_predicate:
            # Set the state for the body inputs, execute the body and collect the outputs.
            # Verify if all outputs are initialized within the branch, split the outputs if they
            # were just passed through, so they can be merged with the other branch.
            with _cond_true():
                body()

                body_state = get_state()

                _verify_branch_outputs(body_state, symbol_names, "if")
                body_outputs = body_state[:nouts]
                body_outputs = apply_conditional_split_to_branch_outputs(body_outputs)

            # Do the same for else block.
            set_state(init_state)
            with _cond_false():
                orelse()

                orelse_state = get_state()

                _verify_branch_outputs(orelse_state, symbol_names, "else")
                orelse_outputs = orelse_state[:nouts]
                orelse_outputs = apply_conditional_split_to_branch_outputs(orelse_outputs)

            # Build the state that is the combination of both branches. Only the actual outputs
            # should be affected by the if/else blocks, the rest can be reused from-before split.
            output_values = []
            # We execute the merge _after_ both branches, and pretend for a moment, that it
            # can see those values produced in child scopes.
            with _cond_merge(split_predicate):
                err_msg = (
                    "Divergent data found in different branches of `if/else` control flow"
                    " statement. Variables in all code paths are merged into common output"
                    " batches. The values assigned to a given variable need to have the same"
                    " nesting structure in every code path (both `if` branches).\n"
                    "For example, if we define a variable as a tuple in one branch, it must"
                    " be defined as a tuple of the same length in the other branch - the"
                    " contents of the tuples may be different. If we define a variable as"
                    " a dictionary, the other branch must define it as a dictionary with the"
                    " same set of keys, the values may be different.\n"
                )

                try:
                    tree.assert_same_structure(body_outputs, orelse_outputs, check_types=True)
                except ValueError as e:
                    # Suppress the original exception, add DALI explanation at the beginning,
                    # raise the full error message.
                    raise ValueError(err_msg + str(e)) from None
                except TypeError as e:
                    raise TypeError(err_msg + str(e)) from None

                def merge_branches(new_body_val, new_orelse_val):
                    logging.log(
                        9,
                        (
                            f"{this_condition_stack()._indent()}[IF] Inserting merge"
                            f" at {this_condition_stack().stack_depth() -1}:"
                            f" merge({new_body_val}, {new_orelse_val}, predicate="
                            f"{split_predicate}."
                        ),
                    )
                    return fn._conditional.merge(
                        new_body_val, new_orelse_val, predicate=split_predicate
                    )

                output_values = _map_structure(merge_branches, body_outputs, orelse_outputs)

        # Register the new nodes outside of the conditional scope, they will be used in subsequent
        # calls.
        this_condition_stack().register_data_nodes(output_values, False)
        # No point in propagating the split/merged values that won't be read later.
        output_values += init_state[nouts:]
        set_state(output_values)

    def detect_overload_not_(self, a):
        return isinstance(a, _DataNode)

    def not_(self, a):
        # Not is eager (not lazy)
        return fn._conditional.not_(a)

    def detect_overload_lazy_and(self, a):
        return isinstance(a, _DataNode)

    def lazy_and(self, a_value, b):
        # We proceed similarly to `if` statement, but we don't have to trace branches and go back.
        # Instead we have one branch already evaluated and conditionally execute the other one.
        # effectively we want `and_output = a_val and b` to be calculated as:
        # if a_val:
        #   and_output = b()
        # else:
        #   and_output = a_val
        a_validated = _validate_logical(a_value, expression_name="and", expression_side="left")
        with _cond_manager(a_validated) as split_predicate:
            with _cond_true():
                b_value = b()
                b_validated = _validate_logical(
                    b_value, expression_name="and", expression_side="right"
                )
                body_outputs = apply_conditional_split(b_validated)
            with _cond_false():
                else_outputs = apply_conditional_split(split_predicate)
            with _cond_merge(split_predicate):
                merged = fn._conditional.merge(
                    body_outputs, else_outputs, predicate=split_predicate
                )

        this_condition_stack().register_data_nodes([merged], False)
        return merged

    def detect_overload_lazy_or(self, a):
        return isinstance(a, _DataNode)

    def lazy_or(self, a_value, b):
        # To implement `or_output = a_val or b` we calculate it as:
        # if a_val:
        #   or_output = a_val
        # else:
        #   or_output = b()
        a_validated = _validate_logical(a_value, expression_name="or", expression_side="left")

        with _cond_manager(a_validated) as split_predicate:
            with _cond_true():
                body_outputs = apply_conditional_split(split_predicate)
            with _cond_false():
                b_value = b()
                b_validated = _validate_logical(
                    b_value, expression_name="or", expression_side="right"
                )
                else_outputs = apply_conditional_split(b_validated)
            with _cond_merge(split_predicate):
                merged = fn._conditional.merge(
                    body_outputs, else_outputs, predicate=split_predicate
                )

        this_condition_stack().register_data_nodes([merged], False)
        return merged


_OVERLOADS = DaliOperatorOverload()

_autograph.initialize_autograph(
    _OVERLOADS,
    convert_modules=["nvidia.dali.auto_aug"],
    do_not_convert_modules=["nvidia.dali._autograph", "nvidia.dali"],
)
