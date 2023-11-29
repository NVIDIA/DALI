# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

import inspect

from nvidia.dali._autograph.utils import hooks


UNSPECIFIED = object()


def overload_of(f):
    if f in SUPPORTED_BUILTINS:
        return BUILTIN_FUNCTIONS_MAP[f.__name__]
    return f


def _find_originating_frame(caller_fn_scope, innermost=True):
    """Locates the frame in which `caller_fn_scope` was defined."""
    ctx_frame = inspect.currentframe()
    result = None
    while ctx_frame is not None:
        # Note it should not be normally possible to get false positives this way
        # because the function scope object is not accessible to user code (barring
        # call stack introspection).
        if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
            result = ctx_frame
            if innermost:
                break
        ctx_frame = ctx_frame.f_back

    assert result is not None, (
        "the conversion process should ensure the caller_fn_scope is always"
        " found somewhere on the call stack"
    )

    return result


def locals_in_original_context(caller_fn_scope):
    """Executes the locals function in the context of a specified function."""
    return _find_originating_frame(caller_fn_scope, innermost=True).f_locals


def globals_in_original_context(caller_fn_scope):
    """Executes the locals function in the context of a specified function."""
    return _find_originating_frame(caller_fn_scope, innermost=True).f_globals


def eval_in_original_context(f, args, caller_fn_scope):
    """Executes the eval function in the context of a specified function."""
    # When control flow is rewritten using functions, eval should use the
    # variables found in the same block where it was called. That is equivalent
    # to the innermost function call.
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)

    args = (
        args[0],
        ctx_frame.f_globals if len(args) < 2 else args[1],
        ctx_frame.f_locals if len(args) < 3 else args[2],
    )
    return f(*args)


def super_in_original_context(f, args, caller_fn_scope):
    """Executes the super function in the context of a specified function.

    See https://docs.python.org/3/library/functions.html#super for the exact
    details

    Args:
      f: Callable, typically the super builtin
      args: List[Any], the original call arguments
      caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
        scope of the converted function in which this call was originally made

    Returns:
      The result of calling `f` as if it was called in the frame indicated by
        `caller_fn_scope`.
    """

    # Only the no-arg call is desugared.
    if args:
        return f(*args)

    # Inner functions seem to include their closure in f_locals, so we need
    # to find the outermost frame.
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)

    # When super(..) is called without arguments, it looks for __class__ cell
    # variable and the first argument passed in the enclosing function according
    # to the spec https://www.python.org/dev/peps/pep-3135/ .
    #
    # We couldn't verify if `inspect.currentframe().f_code.co_varnames[0]` is
    # guaranteed to be the first argument from an official doc or PEP, however,
    # it's fairly stable and well established:
    # - An unofficial community doc mentions it.
    #   https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
    # - CPython has tests checking that order, which was merged in 2008, and
    #   unchanged since then.
    #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py2_test_grammar.py#L157
    #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py3_test_grammar.py#L192
    #
    # Note: the name can be more reliably obtained by inspecting the calling
    # function's argspec.
    #
    # Even though methods can be declared using *args (def method(*args)),
    # that pattern is disallowed by super() -- it raises super() no arguments.
    # Method definitions using **kwargs are not allowed at all.
    # In other words, we can always assume that self is on the first positional
    # argument (for correct code).
    #
    # TODO(mdan): Consider additional checks in case the input code is incorrect.
    # For example, the error might be cryptic compared to what super() regularly
    # raises.

    type_arg = ctx_frame.f_locals["__class__"]
    self_arg_name = ctx_frame.f_code.co_varnames[0]
    self_arg = ctx_frame.f_locals[self_arg_name]
    return f(type_arg, self_arg)


def abs_(x):
    if hooks._DISPATCH.detect_overload_abs_(x):
        return hooks._DISPATCH.abs_(x)
    return _py_abs(x)


def _py_abs(x):
    return abs(x)


def float_(x=0):
    if hooks._DISPATCH.detect_overload_float_(x):
        return hooks._DISPATCH.float_(x)
    return _py_float(x)


def _py_float(x):
    return float(x)


def int_(x=0, base=UNSPECIFIED):
    if hooks._DISPATCH.detect_overload_int_(x):
        return hooks._DISPATCH.int_(x, base)
    return _py_int(x, base)


def _py_int(x, base):
    if base is UNSPECIFIED:
        return int(x)
    return int(x, base)


def len_(s):
    if hooks._DISPATCH.detect_overload_len_(s):
        return hooks._DISPATCH.len_(s)
    return _py_len(s)


def _py_len(s):
    return len(s)


def print_(*objects, **kwargs):
    """Overload of the print builtin."""
    # Note: Python 2.6 doesn't support explicit keywords after starargs.
    unknown_kwargs = tuple(set(kwargs.keys()) - set(("sep", "end", "file", "flush")))
    if unknown_kwargs:
        raise ValueError("invalid keyword arguments: {}".format(unknown_kwargs))
    if hooks._DISPATCH.detect_overload_print_(objects):
        return hooks._DISPATCH.print_(objects, kwargs)
    else:
        _py_print(*objects, **kwargs)


def _py_print(*objects, **kwargs):
    print(*objects, **kwargs)


def min_(*args, **kwargs):
    if hooks._DISPATCH.detect_overload_min_(args):
        return hooks._DISPATCH.min_(*args, **kwargs)
    return _py_min(*args, **kwargs)


def _py_min(*args, **kwargs):
    return min(*args, **kwargs)


def max_(*args, **kwargs):
    if hooks._DISPATCH.detect_overload_max_(args):
        return hooks._DISPATCH.max_(*args, **kwargs)
    return _py_max(*args, **kwargs)


def _py_max(*args, **kwargs):
    return max(*args, **kwargs)


def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
    if hooks._DISPATCH.detect_overload_range_(start_or_stop, stop, step):
        return hooks._DISPATCH.range_(start_or_stop, stop, step)
    return _py_range(start_or_stop, stop, step)


def _py_range(start_or_stop, stop, step):
    if step is not UNSPECIFIED:
        return range(start_or_stop, stop, step)
    if stop is not UNSPECIFIED:
        return range(start_or_stop, stop)
    return range(start_or_stop)


def enumerate_(s, start=0):
    if hooks._DISPATCH.detect_overload_enumerate_(s):
        return hooks._DISPATCH.enumerate_(s, start)
    return _py_enumerate(s, start)


def _py_enumerate(s, start=0):
    return enumerate(s, start)


def zip_(*iterables):
    if hooks._DISPATCH.detect_overload_zip_(iterables):
        return hooks._DISPATCH.zip_(*iterables)
    return _py_zip(*iterables)


def _py_zip(*iterables):
    return zip(*iterables)


def map_(fn, *iterables):
    if hooks._DISPATCH.detect_overload_map_(iterables):
        return hooks._DISPATCH.map_(fn, *iterables)
    return _py_map(fn, *iterables)


def _py_map(fn, *iterables):
    return map(fn, *iterables)


def next_(iterator, default=UNSPECIFIED):
    if hooks._DISPATCH.detect_overload_next_(iterator):
        return hooks._DISPATCH.next_(iterator, default)
    return next_py(iterator, default)


def next_py(iterator, default=UNSPECIFIED):
    if default is UNSPECIFIED:
        return next(iterator)
    return next(iterator, default)


def filter_(function, iterable):
    if hooks._DISPATCH.detect_overload_filter_(iterable):
        return hooks._DISPATCH.filter_(function, iterable)
    return _py_filter(function, iterable)


def _py_filter(function, iterable):
    return filter(function, iterable)


def any_(iterable):
    if hooks._DISPATCH.detect_overload_any_(iterable):
        return hooks._DISPATCH.any_(iterable)
    return _py_any(iterable)


def _py_any(iterable):
    return any(iterable)


def all_(iterable):
    if hooks._DISPATCH.detect_overload_all_(iterable):
        return hooks._DISPATCH.all_(iterable)
    return _py_all(iterable)


def _py_all(iterable):
    return all(iterable)


def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
    if hooks._DISPATCH.detect_overload_sorted_(iterable):
        return hooks._DISPATCH.sorted_(iterable, key, reverse)
    return _py_sorted(iterable, key, reverse)


def _py_sorted(iterable, key, reverse):
    if key is not UNSPECIFIED and reverse is UNSPECIFIED:
        return sorted(iterable, key=key)
    if key is UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, reverse=reverse)
    if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, key=key, reverse=reverse)
    return sorted(iterable)


SUPPORTED_BUILTINS = (
    abs,
    float,
    int,
    len,
    print,
    range,
    enumerate,
    zip,
    map,
    filter,
    any,
    all,
    sorted,
)

BUILTIN_FUNCTIONS_MAP = {
    "abs": abs_,
    "any": any_,
    "all": all_,
    "enumerate": enumerate_,
    "filter": filter_,
    "float": float_,
    "int": int_,
    "len": len_,
    "map": map_,
    "next": next_,
    "print": print_,
    "range": range_,
    "sorted": sorted_,
    "zip": zip_,
}
