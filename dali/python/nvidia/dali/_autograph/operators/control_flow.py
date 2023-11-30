# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Control flow statements: loops, conditionals, etc.

Note: most of these operators accept pairs of get_state/set_state functions, to
capture mutations that the corresponding code blocks might make. These
mutations only need to be captured when staging the control flow, and they just
work when reverting to Python behavior.

__Examples__

```
while cond:
  self.x += i
```

When the functionalized version is executed as a Python loop, it just works:

```
def loop_body():
  self.x += i     # works as expected for Python loops
```

But it won't work for TF loops:

```
def loop_body():
  self.x += i     # self.x has the wrong value!
```

get_state/set_state allow piping the mutations through the loop variables as
well, in effect changing the loop body:

```
def loop_body(self_x):
  self.x = self_x  # self.x now has the proper value
  self.x += i      # the original block
  self_x = self.x  # write self.x back into the loop vars
  return self_x

self_x = tf.while_loop(...)
self.x = self_x    # the result is not properly captured
```
"""

from nvidia.dali._autograph.utils import hooks

# TODO(mdan): Use the custom operator pattern instead of type dispatch.
# An example of this pattern is found in the implementation of distributed
# datasets. Before it can be used though, we need to standardize the interface.


def for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Functional form of a for statement.

    The loop operates on a state, which includes all symbols that are
    variant across loop iterations, excluding the variables local to the loop.

    For example, given the loop below that calculates the geometric and
    arithmetic means or some numbers:

    ```
      geo_mean = 1
      arith_mean = 0
      for i in range(n):
        a = numbers[i]
        geo_mean *= a
        arith_mean += a
    ```

    The state is represented by the variables geo_mean and arith_mean. The
    `extra_test`, `body`, `get_state` and `set_state` functions must bind to the
    original `geo_mean` and `arith_mean` symbols, using `nonlocal`.

    The inputs and outputs of the callables representing the loop blocks are not
    explicit - instead, these functions must use nonlocal/global for side effects.
    The inputs and outputs are instead controlled by the set_state/get_state
    functions.

    Args:
      iter_: The entity being iterated over.
      extra_test: Callable with boolean return type. An additional loop condition.
      body: Callable representing the actual loop body.
      get_state: Additional callable which can capture additional state (such as
        the values of composite symbols). This is only useful when staging the
        loop.
      set_state: Additional callable which save values captured by get_state back
        into the Python environment. This is only useful when staging the loop.
      symbol_names: Tuple containing names of the loop variables returned by
        get_state.
      opts: Optional dict of extra loop parameters.
    """
    if hooks._DISPATCH.detect_overload_for_stmt(iter_):
        hooks._DISPATCH.for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts)
    else:
        _py_for_stmt(iter_, extra_test, body, None, None)


def _py_for_stmt(iter_, extra_test, body, get_state, set_state):
    """Overload of for_stmt that executes a Python for loop."""
    del get_state, set_state

    if extra_test is not None:
        if extra_test():
            for target in iter_:
                body(target)
                if not extra_test():
                    break

    else:
        for target in iter_:
            body(target)


def while_stmt(test, body, get_state, set_state, symbol_names, opts):
    """Functional form of a while statement.

    The loop operates on a so-called state, which includes all symbols that are
    variant across loop iterations. In what follows we refer to state as either
    a tuple of entities that represent an actual state, or a list of arguments
    of the corresponding types.

    The inputs and outputs of the callables representing the loop blocks are not
    explicit - instead, these functions must use nonlocal/global for side effects.
    The inputs and outputs are instead controlled by the set_state/get_state
    functions.

    Args:
      test: Callable with boolean return type. The loop condition.
      body: Callable representing the actual loop body.
      get_state: Additional callable which can capture additional state (such as
        the values of composite symbols). This is only useful when staging the
        loop.
      set_state: Additional callable which save values captured by get_state back
        into the Python environment. This is only useful when staging the loop.
      symbol_names: Tuple containing the names of all loop variables.
      opts: Optional dict of extra loop parameters.

    Returns:
      Tuple containing the final state.
    """

    # Evaluate the initial test once in order to do the dispatch. The evaluation
    # is isolated to minimize unwanted side effects.
    # TODO(mdan): Do a full iteration - some state types might lower to Tensor.
    # with func_graph.FuncGraph('tmp').as_default():
    #   init_test = test()

    init_test = test()

    # TensorFlow: Multiple evaluations are acceptable in this case, so we're fine
    # with the re-evaluation of `test` that `_tf_while_stmt` will make.
    if hooks._DISPATCH.detect_overload_while_stmt(test):
        hooks._DISPATCH.while_stmt(test, body, get_state, set_state, symbol_names, opts)
        return

    # Normal Python: We already consumed one evaluation of `test`; consistently,
    # unroll one iteration before dispatching to a normal loop.
    # TODO(mdan): Push the "init_test" value via opts into _py_while_stmt?
    if not init_test:
        return
    body()

    _py_while_stmt(test, body, get_state, set_state, opts)


def _py_while_stmt(test, body, get_state, set_state, opts):
    """Overload of while_stmt that executes a Python while loop."""
    del opts, get_state, set_state

    while test():
        body()


def if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts):
    """Functional form of an if statement.

    The conditional operates on a state, which includes all symbols whose values
    are a function of the branch taken.

    For example, given the code below that calculates the abs function:

    ```
      x = 1
      if x > 0:
        x = -x
    ```

    The state is represented by the variable `x`. The `body, `orelse` and
    `set_state` functions must bind to the original `x` symbol, using `nonlocal`.

    The inputs and outputs of the callables representing the loop blocks are not
    explicit - instead, these functions must use nonlocal/global for side effects.
    The inputs and outputs are instead controlled by the set_state/get_state
    functions.

    Args:
      cond: Boolean.
      body: Callable representing the main block of the conditional.
      orelse: Callable representing the else block of the conditional.
      get_state: Function that returns a tuple containing the values of all
        composite symbols modified within the conditional. This allows access to
        state that branches may mutate through side effects. This function is not
        needed and should not be called when dispatching to code matching Python's
        default semantics. This is useful for checkpointing to avoid unintended
        side-effects when staging requires evaluating all code-paths.
      set_state: Function to set the values of all composite symbols modified
        within the conditional. This is the complement to get_state, used to
        restore checkpointed values. The single argument a tuple containing values
        for each composite symbol that may be modified in a branch of the
        conditional. The is usually the result of a call to get_state.
      symbol_names: Tuple containing basic loop var names.
      nouts: Number of variables output by the statement. Vars which are not
        outputs will not be passed through staged control flow such as tf.cond.
        This includes variables that are defined before the conditional, but are
        not used after it.
    """
    # Note: tf.cond doesn't support SparseTensor.
    if hooks._DISPATCH.detect_overload_if_stmt(cond):
        return hooks._DISPATCH.if_stmt(
            cond, body, orelse, get_state, set_state, symbol_names, nouts
        )
    else:
        _py_if_stmt(cond, body, orelse)


def _py_if_stmt(cond, body, orelse):
    """Overload of if_stmt that executes a Python if statement."""
    return body() if cond else orelse()
