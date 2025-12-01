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
"""Generic utilities not strictly related to autograph that are moved here or just implemented
as no-op placeholders if the actual functionality doesn't matter - for example the scope of API
export and it's management is not important if we import the autograph as internal symbol.
"""

import inspect


def _remove_undocumented(module_name, allowed_exception_list=None, doc_string_modules=None):
    pass


def export_symbol(*args, **kwargs):
    """No-op replacement for @tf_export. This is decorator factory that accepts arguments"""

    def actual_decorator(function):
        return function

    return actual_decorator


def make_decorator(
    target, decorator_func, decorator_name=None, decorator_doc="", decorator_argspec=None
):
    """Make a decorator from a wrapper and a target.

    Args:
      target: The final callable to be wrapped.
      decorator_func: The wrapper function.
      decorator_name: The name of the decorator. If `None`, the name of the
        function calling make_decorator.
      decorator_doc: Documentation specific to this application of
        `decorator_func` to `target`.
      decorator_argspec: The new callable signature of this decorator.

    Returns:
      The `decorator_func` argument with new metadata attached.
      Note that we just wrap the function and adjust the members but do not insert the special
      member TFDecorator
    """
    if decorator_name is None:
        decorator_name = inspect.currentframe().f_back.f_code.co_name
    # Objects that are callables (e.g., a functools.partial object) may not have
    # the following attributes.
    if hasattr(target, "__name__"):
        decorator_func.__name__ = target.__name__
    if hasattr(target, "__qualname__"):
        decorator_func.__qualname__ = target.__qualname__
    if hasattr(target, "__module__"):
        decorator_func.__module__ = target.__module__
    if hasattr(target, "__dict__"):
        # Copy dict entries from target which are not overridden by decorator_func.
        for name in target.__dict__:
            if name not in decorator_func.__dict__:
                decorator_func.__dict__[name] = target.__dict__[name]
    decorator_func.__wrapped__ = target
    # Keeping a second handle to `target` allows callers to detect whether the
    # decorator was modified using `rewrap`.
    decorator_func.__original_wrapped__ = target
    return decorator_func


# TODO(klecki): Introduce tests for control flow of integrated library (DALI)
def custom_constant(val, shape=None, dtype=None):
    """Customization point to introduce library-specific argument to the control flow.
    Currently those tests fallback to Python implementation"""
    return val
