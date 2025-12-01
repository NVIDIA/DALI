# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class OperatorBase:
    """User may implement hooks for detection of overloads that are inserted by autograph
    and their implementation.

    In TF's AutoGraph, the AST is transformed to insert AutoGraph operators (operators/ dir),
    for example if and for statements are replaced by ag__.if_stmt and ag__.for_stmt function
    calls.
    In such operator AutoGraph checks if one of the arguments is the user-defined type of interest
    (by default a tf.Tensor of tf.data.Dataset) and if so, appropriate graph operations are
    inserted. Otherwise a _py... fallback like _py_if_stmt or _py_for_stmt is invoked, providing
    default Python semantics.

    The user that wants to customize AutoGraph can do so, by overloading OperatorBase.
    Any `detect_overload_x` function is used to detect the objects of user-defined type,
    corresponding `x` function is supposed to implement the particular overload.
    If `detect_overload_x` returns True, the `x` is called.

    For example, `detect_overload_for_stmt` can be used to recognize that the iter_ in for statement
    is a custom user-defined type and the iteration should have a custom implementation,
    that is provided in `for_stmt`.

    See the documentation in operators/ for description of the arguments used in those overloads,
    the names of the functions are matching, so OperatorBase.for_stmt <-> control_flow.for_stmt.
    """

    def detect_overload(self, object):
        """Generic detection of custom user-defined type used for all overloads.

        Parameters
        ----------
        object
            Custom object detected by user or regular Python type.

        Returns
        -------
        bool
            True if custom operator implementation should be always used, False to fallback to
            Python behavior.
        """
        return False

    def detect_overload_ld(self, v):
        return self.detect_overload(v)

    def ld(self, v):
        pass

    def detect_overload_if_exp(self, cond):
        return self.detect_overload(cond)

    def if_exp(self, cond, if_true, if_false, expr_repr):
        pass

    def detect_overload_for_stmt(self, iter_):
        return self.detect_overload(iter_)

    def for_stmt(self, iter_, extra_test, body, get_state, set_state, symbol_names, opts):
        pass

    def detect_overload_while_stmt(self, test):
        return self.detect_overload(test)

    def while_stmt(self, test, body, get_state, set_state, symbol_names, opts):
        pass

    def detect_overload_if_stmt(self, cond):
        return self.detect_overload(cond)

    def if_stmt(self, cond, body, orelse, get_state, set_state, symbol_names, nouts):
        pass

    def detect_overload_assert_stmt(self, expression1):
        return self.detect_overload(expression1)

    def assert_stmt(self, expression1, expression2):
        pass

    def detect_overload_not_(self, a):
        return self.detect_overload(a)

    def not_(self, a):
        pass

    def detect_overload_lazy_and(self, a):
        return self.detect_overload(a)

    def lazy_and(self, a_val, b):
        pass

    def detect_overload_lazy_or(self, a):
        return self.detect_overload(a)

    def lazy_or(self, a_val, b):
        pass

    def detect_overload_equal(self, a):
        return self.detect_overload(a)

    def equal(self, a, b):
        pass

    def detect_overload_abs_(self, a):
        return self.detect_overload(a)

    def abs_(self, x):
        pass

    def detect_overload_float_(self, x):
        return self.detect_overload(x)

    def float_(self, x):
        pass

    def detect_overload_int_(self, x):
        return self.detect_overload(x)

    def int_(self, x, base):
        pass

    def detect_overload_len_(self, x):
        return self.detect_overload(x)

    def len_(self, s):
        pass

    def detect_overload_print_(self, objects):
        return any(self.detect_overload(x) for x in objects)

    def print_(self, objects, kwargs):
        pass

    def detect_overload_min_(self, args):
        return any(self.detect_overload(x) for x in args)

    def min_(self, *args, **kwargs):
        pass

    def detect_overload_max_(self, args):
        return any(self.detect_overload(x) for x in args)

    def max_(self, *args, **kwargs):
        pass

    def detect_overload_range_(self, start_or_stop, stop, step):
        return any(self.detect_overload(x) for x in (start_or_stop, stop, step))

    def range_(self, start_or_stop, stop, step):
        pass

    def detect_overload_enumerate_(self, s):
        return self.detect_overload(s)

    def enumerate_(self, s, start):
        pass

    def detect_overload_zip_(self, iterables):
        return all(self.detect_overload(x) for x in iterables)

    def zip_(self, *iterables):
        pass

    def detect_overload_map_(self, iterables):
        return all(self.detect_overload(x) for x in iterables)

    def map_(self, fn, *iterables):
        pass

    def detect_overload_next_(self, iterator):
        return self.detect_overload(iterator)

    def next_(self, iterator, default):
        pass

    def detect_overload_filter_(self, iterable):
        return self.detect_overload(iterable)

    def filter_(self, function, iterable):
        pass

    def detect_overload_any_(self, iterable):
        return self.detect_overload(iterable)

    def any_(self, iterable):
        pass

    def detect_overload_all_(self, iterable):
        return self.detect_overload(iterable)

    def all_(self, iterable):
        pass

    def detect_overload_sorted_(self, iterable):
        return self.detect_overload(iterable)

    def sorted_(self, iterable, key, reverse):
        pass

    def detect_overload_get_item(self, target):
        return self.detect_overload(target)

    def get_item(self, target, i):
        pass

    def detect_overload_set_item(self, target):
        return self.detect_overload(target)

    def set_item(self, target, i, x):
        pass

    def detect_overload_list_new(self, iterable):
        return self.detect_overload(iterable)

    def list_new(self, iterable):
        pass

    def detect_overload_list_append(self, list_):
        return self.detect_overload(list_)

    def list_append(self, list_, x):
        pass

    def detect_overload_list_pop(self, list_):
        return self.detect_overload(list_)

    def list_pop(self, list_, i):
        pass

    def detect_overload_list_stack(self, list_):
        return self.detect_overload(list_)

    def list_stack(self, list_, opts):
        pass


_DISPATCH = OperatorBase()
