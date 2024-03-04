# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""This module contains the user- and codegen-facing API for AutoGraph."""

import functools
import importlib
import inspect
import os
import sys
import textwrap
import traceback

from nvidia.dali._autograph import operators
from nvidia.dali._autograph import utils
from nvidia.dali._autograph.converters import asserts
from nvidia.dali._autograph.converters import break_statements
from nvidia.dali._autograph.converters import call_trees
from nvidia.dali._autograph.converters import conditional_expressions
from nvidia.dali._autograph.converters import continue_statements
from nvidia.dali._autograph.converters import control_flow
from nvidia.dali._autograph.converters import directives
from nvidia.dali._autograph.converters import functions
from nvidia.dali._autograph.converters import lists
from nvidia.dali._autograph.converters import logical_expressions
from nvidia.dali._autograph.converters import return_statements
from nvidia.dali._autograph.converters import slices
from nvidia.dali._autograph.converters import variables
from nvidia.dali._autograph.core import ag_ctx
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.core import config
from nvidia.dali._autograph.core import function_wrappers
from nvidia.dali._autograph.core import unsupported_features_checker
from nvidia.dali._autograph.impl import conversion
from nvidia.dali._autograph.operators import py_builtins
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import error_utils
from nvidia.dali._autograph.pyct import errors
from nvidia.dali._autograph.pyct import inspect_utils
from nvidia.dali._autograph.pyct import origin_info
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transpiler
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import reaching_definitions
from nvidia.dali._autograph.utils import hooks
from nvidia.dali._autograph.utils import ag_logging as logging
from nvidia.dali._autograph.utils import all_utils

from nvidia.dali._autograph.utils import tf_stack
from nvidia.dali._autograph.utils.all_utils import export_symbol


def is_autograph_strict_conversion_mode():
    return int(os.environ.get("AUTOGRAPH_STRICT_CONVERSION", "0")) > 0


#
# Error handling
#


# TODO(mdan): Export this symbol.
class AutoGraphError(errors.PyCTError):
    """Base class for all AutoGraph exceptions."""

    pass


class ConversionError(AutoGraphError):
    """Raised during the conversion process."""

    pass


class StagingError(AutoGraphError):
    """Raised during the staging (i.e. Python execution) of converted code."""

    pass


class _ErrorMetadata(error_utils.ErrorMetadataBase):
    """AutoGraph-specific error metadata. See base class."""

    def create_exception(self, source_error):
        preferred_type = type(source_error)
        if preferred_type in (errors.PyCTError, AutoGraphError, ConversionError, StagingError):
            return preferred_type(self.get_message())

        exc = super(_ErrorMetadata, self).create_exception(source_error)
        if exc is not None:
            return exc

        # Note: While changing an error's message property to change the message it
        # displays will probably work a lot of times, there is no standard way in
        # Python to do that. The safest way is therefore to create a new exception.
        # For user defined exceptions, we could define an interface that allowed
        # them to work under this mechanism.
        return StagingError(self.get_message())


def _attach_error_metadata(e, f):
    """Augments an error with the metadata necessary for rewrite."""
    if hasattr(e, "ag_pass_through"):
        return

    metadata = getattr(e, "ag_error_metadata", None)
    source_map = f.ag_source_map

    if metadata is None:
        logging.log(1, "Caught error in user callable %s", f, exc_info=True)
        message = "{}: {}".format(e.__class__.__name__, e)
    else:
        message = None

    cause_tb = traceback.extract_tb(sys.exc_info()[2])[1:]

    e.ag_error_metadata = _ErrorMetadata(cause_tb, metadata, message, source_map, __file__)


class StackTraceMapper(tf_stack.StackTraceMapper):
    """Remaps generated code to code it originated from."""

    def __init__(self, converted_fn):
        super().__init__()
        self._source_map = converted_fn.ag_source_map
        # This may be called repeatedly: once on entry, by the superclass, then by
        # each child context manager.
        self._cached_map = None

    def get_effective_source_map(self):
        if self._cached_map is not None:
            return self._cached_map

        parent_map = self.parent.get_effective_source_map()

        effective_source_map = {}
        for loc, origin in self._source_map.items():
            effective_source_map[(loc.filename, loc.lineno)] = (
                origin.loc.filename,
                origin.loc.lineno,
                origin.function_name,
                origin.source_code_line,
            )

        for key, value in parent_map.items():
            filename, lineno, _, _ = value
            value_loc = origin_info.LineLocation(filename=filename, lineno=lineno)
            if value_loc in self._source_map:
                origin = self._source_map[value_loc]
                effective_source_map[key] = (
                    origin.loc.filename,
                    origin.loc.lineno,
                    origin.function_name,
                    origin.source_code_line,
                )
            else:
                effective_source_map[key] = value

        self._cached_map = effective_source_map
        return effective_source_map


#
# Actual source code transformation
#


class PyToLib(transpiler.PyToPy):
    """The TensorFlow AutoGraph transformer."""

    def __init__(self, name, operator_overload):
        super(PyToLib, self).__init__()
        self._name = name
        self._operator_overload = operator_overload
        self._extra_locals = None

    def get_transformed_name(self, node):
        return self._name + "__" + super(PyToLib, self).get_transformed_name(node)

    def get_extra_locals(self):
        if self._extra_locals is None:
            # TODO(mdan): Move into core or replace with an actual importable module.
            # Craft a module that exposes the external API as well as certain
            # internal modules.
            module_spec = importlib.machinery.ModuleSpec(self._name, None)
            ag_internal = importlib.util.module_from_spec(module_spec)
            ag_internal.__dict__.update(inspect.getmodule(PyToLib).__dict__)
            ag_internal.ConversionOptions = converter.ConversionOptions
            ag_internal.STD = converter.STANDARD_OPTIONS
            ag_internal.Feature = converter.Feature
            ag_internal.utils = utils
            ag_internal.FunctionScope = function_wrappers.FunctionScope
            ag_internal.with_function_scope = function_wrappers.with_function_scope
            # TODO(mdan): Add safeguards against name clashes.
            # We don't want to create a submodule because we want the operators to be
            # accessible as ag__.<operator>
            ag_internal.__dict__.update(operators.__dict__)
            ag_internal.hooks = hooks
            ag_internal.hooks._DISPATCH = self._operator_overload

            self._extra_locals = {"ag__": ag_internal}
        return self._extra_locals

    def get_caching_key(self, ctx):
        return ctx.options

    def initial_analysis(self, node, ctx):
        graphs = cfg.build(node)
        node = qual_names.resolve(node)
        node = activity.resolve(node, ctx, None)
        node = reaching_definitions.resolve(node, ctx, graphs)
        anno.dup(
            node,
            {
                anno.Static.DEFINITIONS: anno.Static.ORIG_DEFINITIONS,
            },
        )
        return node

    def transform_ast(self, node, ctx):
        unsupported_features_checker.verify(node)
        node = self.initial_analysis(node, ctx)

        node = functions.transform(node, ctx)
        node = directives.transform(node, ctx)
        node = break_statements.transform(node, ctx)
        if ctx.user.options.uses(converter.Feature.ASSERT_STATEMENTS):
            node = asserts.transform(node, ctx)
        # Note: sequencing continue canonicalization before for loop one avoids
        # dealing with the extra loop increment operation that the for
        # canonicalization creates.
        node = continue_statements.transform(node, ctx)
        node = return_statements.transform(node, ctx)
        if ctx.user.options.uses(converter.Feature.LISTS):
            node = lists.transform(node, ctx)
            node = slices.transform(node, ctx)
        node = call_trees.transform(node, ctx)
        node = control_flow.transform(node, ctx)
        node = conditional_expressions.transform(node, ctx)
        node = logical_expressions.transform(node, ctx)
        node = variables.transform(node, ctx)
        return node


def _convert_actual(entity, program_ctx):
    """Applies AutoGraph to entity."""

    # TODO(mdan): Put these extra fields inside __autograph_info__.
    if not hasattr(entity, "__code__"):
        raise ValueError(
            "Cannot apply autograph to a function that doesn't " "expose a __code__ object."
        )

    transformed, module, source_map = _TRANSPILER.transform(entity, program_ctx)

    assert not hasattr(transformed, "ag_module")
    assert not hasattr(transformed, "ag_source_map")
    transformed.ag_module = module
    transformed.ag_source_map = source_map
    return transformed


#
# Generated code support
#


def autograph_artifact(entity, extras=None):
    if inspect.ismethod(entity):
        setattr(entity.__func__, "autograph_info__", extras)
    else:
        setattr(entity, "autograph_info__", extras)
    return entity


def is_autograph_artifact(entity):
    return hasattr(entity, "autograph_info__")


def is_frame_ag_call_entrypoint(frame_info):
    """
    True if the given frame is start of a function call wrapped by AutoGraph (ag__.converted_call)
    """
    return (
        frame_info.filename.endswith("nvidia/dali/_autograph/impl/api.py")
        and frame_info.name == "converted_call"
    )


def is_frame_ag_call_unconverted(frame_info):
    """True if the given frame exits autograph to call unconverted user code."""
    return (
        frame_info.filename.endswith("nvidia/dali/_autograph/impl/api.py")
        and frame_info.name == "_call_unconverted"
    )


def converted_call(f, args, kwargs, caller_fn_scope=None, options=None):
    """Converts a function call inline.

    For internal use only.

    Note: The argument list is optimized for readability of generated code, which
    may look like this:

      ag__.converted_call(f, (arg1, arg2), None, fscope)
      ag__.converted_call(f, (), dict(arg1=val1, **kwargs), fscope)
      ag__.converted_call(f, (arg1, arg2) + varargs, dict(**kwargs), lscope)

    Args:
      f: The function to convert.
      args: Tuple, the original positional arguments of f
      kwargs: Optional[Dict], the original keyword arguments of f
      caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
        scope of the converted function in which this call was originally made.
      options: Optional[converter.ConversionOptions], conversion options. If not
        specified, the value of caller_fn_scope.callopts is used. Either options
        or caller_fn_scope must be present.

    Returns:
      Any, the result of executing a possibly-converted `f` with the given
        arguments.
    """
    logging.log(1, "Converted call: %s\n    args: %s\n    kwargs: %s\n", f, args, kwargs)

    if options is None:
        if caller_fn_scope is None:
            raise ValueError("either caller_fn_scope or options must have a value")
        options = caller_fn_scope.callopts

    if conversion.is_in_allowlist_cache(f, options):
        logging.log(2, "Allowlisted %s: from cache", f)
        return _call_unconverted(f, args, kwargs, options, False)

    if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:
        logging.log(2, "Allowlisted: %s: AutoGraph is disabled in context", f)
        return _call_unconverted(f, args, kwargs, options, False)

    if is_autograph_artifact(f):
        logging.log(2, "Permanently allowed: %s: AutoGraph artifact", f)
        return _call_unconverted(f, args, kwargs, options)

    # If this is a partial, unwrap it and redo all the checks.
    if isinstance(f, functools.partial):
        new_kwargs = {}
        if f.keywords is not None:
            # Use copy to avoid mutating the underlying keywords.
            new_kwargs = f.keywords.copy()
        if kwargs is not None:
            new_kwargs.update(kwargs)
        new_args = f.args + args
        logging.log(3, "Forwarding call of partial %s with\n%s\n%s\n", f, new_args, new_kwargs)
        return converted_call(
            f.func, new_args, new_kwargs, caller_fn_scope=caller_fn_scope, options=options
        )

    if inspect_utils.isbuiltin(f):
        if f is eval:
            return py_builtins.eval_in_original_context(f, args, caller_fn_scope)
        if f is super:
            return py_builtins.super_in_original_context(f, args, caller_fn_scope)
        if f is globals:
            return py_builtins.globals_in_original_context(caller_fn_scope)
        if f is locals:
            return py_builtins.locals_in_original_context(caller_fn_scope)
        if kwargs:
            return py_builtins.overload_of(f)(*args, **kwargs)
        else:
            return py_builtins.overload_of(f)(*args)

    if conversion.is_unsupported(f):
        return _call_unconverted(f, args, kwargs, options)

    if not options.user_requested and conversion.is_allowlisted(f):
        return _call_unconverted(f, args, kwargs, options)

    # internal_convert_user_code is for example turned off when issuing a dynamic
    # call conversion from generated code while in nonrecursive mode. In that
    # case we evidently don't want to recurse, but we still have to convert
    # things like builtins.
    if not options.internal_convert_user_code:
        return _call_unconverted(f, args, kwargs, options)

    try:
        if inspect.ismethod(f) or inspect.isfunction(f):
            target_entity = f
            effective_args = args

            f_self = getattr(f, "__self__", None)
            if f_self is not None:
                effective_args = (f_self,) + effective_args

        elif hasattr(f, "__class__") and hasattr(f.__class__, "__call__"):
            # Callable objects. Dunder methods have special lookup rules, see:
            # https://docs.python.org/3/reference/datamodel.html#specialnames
            # TODO(mdan): Recurse into converted_call to simplify other verifications.
            # This should be handled in the same way as partials.
            target_entity = f.__class__.__call__
            effective_args = (f,) + args

        else:
            target_entity = f
            raise NotImplementedError('unknown callable type "%s"' % type(f))

    except Exception as e:  # pylint:disable=broad-except
        logging.log(1, "Error transforming entity %s", target_entity, exc_info=True)
        if is_autograph_strict_conversion_mode():
            raise
        return _fall_back_unconverted(f, args, kwargs, options, e)

    if not hasattr(target_entity, "__code__"):
        logging.log(2, "Permanently allowed: %s: native binding", target_entity)
        return _call_unconverted(f, args, kwargs, options)
    elif (
        hasattr(target_entity.__code__, "co_filename")
        and target_entity.__code__.co_filename == "<string>"
    ):
        # TODO(mdan): __globals__['txt'] might work in Py3.
        logging.log(2, "Permanently allowed: %s: dynamic code (exec?)", target_entity)
        return _call_unconverted(f, args, kwargs, options)

    try:
        program_ctx = converter.ProgramContext(options=options)
        converted_f = _convert_actual(target_entity, program_ctx)
        if logging.has_verbosity(2):
            _log_callargs(converted_f, effective_args, kwargs)
    except Exception as e:  # pylint:disable=broad-except
        logging.log(1, "Error transforming entity %s", target_entity, exc_info=True)
        if is_autograph_strict_conversion_mode():
            raise
        return _fall_back_unconverted(f, args, kwargs, options, e)

    # We no longer need CurrentModuleFilter here, as we filter whole autograph
    # TODO(klecki): Filter them just once.
    import nvidia.dali._conditionals as dc
    import nvidia.dali._autograph as ag

    with StackTraceMapper(converted_f), tf_stack.CustomModuleFilter([ag, dc]):
        try:
            if kwargs is not None:
                result = converted_f(*effective_args, **kwargs)
            else:
                result = converted_f(*effective_args)
        except Exception as e:
            _attach_error_metadata(e, converted_f)
            raise

    return result


def _call_unconverted(f, args, kwargs, options, update_cache=True):
    """Calls the original function without converting with AutoGraph."""
    if update_cache:
        conversion.cache_allowlisted(f, options)

    if kwargs is not None:
        return f(*args, **kwargs)
    return f(*args)


def _fall_back_unconverted(f, args, kwargs, options, exc):
    """Falls back to calling the function unconverted, in case of error."""
    # TODO(mdan): Consider adding an internal metric.
    warning_template = (
        "AutoGraph could not transform %s and will run it as-is.\n"
        "%s"
        "Cause: %s\n"
        "To silence this warning, decorate the function with @nvidia.dali.pipeline.do_not_convert"
    )
    if isinstance(exc, errors.InaccessibleSourceCodeError):
        if ag_ctx.INSPECT_SOURCE_SUPPORTED:
            logging.warning(warning_template, f, "", exc)
    elif isinstance(exc, errors.UnsupportedLanguageElementError):
        if not conversion.is_in_allowlist_cache(f, options):
            logging.warning(warning_template, f, "", exc)
    else:
        # TODO(klecki): Do we want to report such errors?
        # file_bug_message = (
        #     'Please report this to the TensorFlow team. When filing the bug, set'
        #     ' the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and'
        #     ' attach the full output.\n')
        logging.warning(warning_template, f, "", exc)

    return _call_unconverted(f, args, kwargs, options)


#
# TensorFlow integration
#


@export_symbol("__internal__.autograph.tf_convert", v1=[])
def tf_convert(f, ctx, convert_by_default=True, user_requested=False):
    """Decorator that applies AutoGraph to a function.

    Use in internal APIs.

    This API is suitable for high order functions internal to the TensorFlow API,
    and more generally any function to which AutoGraph is not applied.

    Guidance: `convert` was a decorator meant for use directly by developers, but
    most of today's uses go through `tf.function`. `tf_convert` is to be called
    from high order functions internal to TF. By default, all the internal
    TensorFlow functions are skipped when AutoGraph processes the code. This may
    lead to user-supplied functions to be incorrectly skipped as well.
    `tf_convert` helps avoid that. See the following example for more details.

    ```
    =====tf_internal_module.py=====

    def unconverted(input_fn):
      return input_fn()

    def converted(input_fn):
      return tf.__internal__.autograph.tf_convert(
         input_fn, ctx=tf.__internal__.autograph.control_status_ctx())()

    ======user_module.py======

    @tf.function
    def foo(input_fn)
      return unconverted(input_fn)

    @tf.function
    def bar(input_fn)
      return converted(input_fn)

    @tf.function(autograph=False)
    def baz(input_fn)
      return converted(input_fn)
    ```

    The `foo` method above will execute the `input_fn` without autograph
    conversion, while the `bar` method will run an autographed `input_fn`. The
    `baz` method will run an unconverted `input_fn`, since `tf_convert` respect
    the control status context.

    Note that both methods in `tf_internal_module` are skipped by autograph when
    tracing the `tf.function`. The configuration of whether a module/package
    should be skipped by autograph is controlled in
    tensorflow/python/autograph/core/config.py.

    Args:
      f: Callable.
      ctx: ag_ctx.ControlStatusCtx, the Autograph context in which `f` is used.
      convert_by_default: bool, whether to use AutoGraph when the context doesn't
        specify.
      user_requested: bool, whether to ignore the conversion allowlist. See
        ConversionOptions.user_requested.

    Returns:
      Either `f or the converted version of `f`.
    """

    if is_autograph_artifact(f):
        return f

    # TODO(mdan): Grab features from context.
    # Note: we pass the original context through to convert to properly handle the
    # following scenario, which can be used inside TF implementations:
    #
    #   ctx = ag_ctx.control_status_ctx()
    #   @function(autograph=False)  # Low-level graph code
    #   def inner_fn():
    #     # The context is disabled here, but should be enabled in user user_fn
    #     tf_convert(user_fn, ctx=ctx)
    if ctx.status == ag_ctx.Status.ENABLED:
        wrapper_factory = convert(recursive=True, user_requested=user_requested, conversion_ctx=ctx)
    elif ctx.status == ag_ctx.Status.DISABLED:
        wrapper_factory = do_not_convert
    elif ctx.status == ag_ctx.Status.UNSPECIFIED:
        if convert_by_default:
            wrapper_factory = convert(
                recursive=True, user_requested=user_requested, conversion_ctx=ctx
            )
        else:
            wrapper_factory = call_with_unspecified_conversion_status
    else:
        assert False, "This switch contains all possible cases!"
    wrapper = wrapper_factory(f)

    return autograph_artifact(wrapper)


def call_with_unspecified_conversion_status(func):
    """Decorator that resets the conversion context to the unspecified status."""

    def wrapper(*args, **kwargs):
        with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.UNSPECIFIED):
            return func(*args, **kwargs)

    if inspect.isfunction(func) or inspect.ismethod(func):
        wrapper = functools.update_wrapper(wrapper, func)

    return autograph_artifact(wrapper)


def _log_callargs(f, args, kwargs):
    """Logging helper."""
    logging.log(2, "Defaults of %s : %s", f, f.__defaults__)
    logging.log(2, "KW defaults of %s : %s", f, f.__kwdefaults__)

    if kwargs is not None:
        callargs = inspect.getcallargs(f, *args, **kwargs)
    else:
        callargs = inspect.getcallargs(f, *args)

    formatted_callargs = "\n".join("    {}: {}".format(k, v) for k, v in callargs.items())
    logging.log(2, "Calling %s with\n%s\n", f, formatted_callargs)


#
# Public API
#


def do_not_convert(func=None):
    """Decorator that suppresses the conversion of a function.

    Args:
      func: function to decorate.

    Returns:
      If `func` is not None, returns a `Callable` which is equivalent to
      `func`, but is not converted by AutoGraph.
      If `func` is None, returns a decorator that, when invoked with a
      single `func` argument, returns a `Callable` equivalent to the
      above case.
    """
    if func is None:
        return do_not_convert

    def wrapper(*args, **kwargs):
        with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
            return func(*args, **kwargs)

    if inspect.isfunction(func) or inspect.ismethod(func):
        wrapper = functools.update_wrapper(wrapper, func)

    return autograph_artifact(wrapper)


# TODO(mdan): Make private.
def convert(
    recursive=False, optional_features=None, user_requested=True, conversion_ctx=ag_ctx.NullCtx()
):
    """Decorator that compiles a function to use TensorFlow ops.

    The decorator is dynamic - it recompiles the target whenever the decorated
    function is called. This means the parameter values are known at conversion.
    It also means that repeated calls with different types of parameters will be
    correctly processed.

    Args:
      recursive: bool, whether to recursively convert any functions or classes
        that the converted function may use.
      optional_features: converted.Feature, allows toggling optional or
        experimental features. When set to None, only the core features are
        enabled.
      user_requested: bool, whether this is a function that the user explicitly
        asked to be converted. See ConversionOptions.user_requested.
      conversion_ctx: Optional ag_ctx.ControlStatusCtx, the Autograph context in
        which `f` is used.

    Returns:
      Callable, a decorator that converts the given function into an equivalent
      function that uses TensorFlow ops.
    """

    def decorator(f):
        """Decorator implementation."""

        def wrapper(*args, **kwargs):
            """Wrapper that calls the converted version of f."""
            options = converter.ConversionOptions(
                recursive=recursive,
                user_requested=user_requested,
                optional_features=optional_features,
            )
            try:
                with conversion_ctx:
                    return converted_call(f, args, kwargs, options=options)
            except Exception as e:  # pylint:disable=broad-except
                if hasattr(e, "ag_error_metadata"):
                    raise e.ag_error_metadata.to_exception(e)
                else:
                    raise

        if inspect.isfunction(f) or inspect.ismethod(f):
            wrapper = functools.update_wrapper(wrapper, f)

        decorated_wrapper = all_utils.make_decorator(f, wrapper)
        return autograph_artifact(decorated_wrapper)

    return decorator


# pylint:disable=line-too-long
@export_symbol("autograph.to_graph", v1=[])
def to_graph(entity, recursive=True, experimental_optional_features=None):
    """Converts a Python entity into a TensorFlow graph.

      Also see: `tf.autograph.to_code`, `tf.function`.

      Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
      Python code to TensorFlow graph code. It does not implement any caching,
      variable management or create any actual ops, and is best used where greater
      control over the generated TensorFlow graph is desired. Another difference
      from `tf.function` is that `to_graph` will not wrap the graph into a
      TensorFlow function or a Python callable. Internally, `tf.function` uses
      `to_graph`.

      Example usage:

      >>> def f(x):
      ...   if x > 0:
      ...     y = x * x
      ...   else:
      ...     y = -x
      ...   return y
      ...
      >>> converted_f = to_graph(f)
      >>> x = tf.constant(2)
      >>> converted_f(x)  # converted_foo is like a TensorFlow Op.
      <tf.Tensor: shape=(), dtype=int32, numpy=4>

      Supported Python entities include:
        * functions
        * classes
        * object methods

      Functions are converted into new functions with converted code.

      Classes are converted by generating a new class whose methods use converted
      code.

      Methods are converted into unbound function that have an additional first
      argument called `self`.

      For a tutorial, see the
      [tf.function and AutoGraph guide](https://www.tensorflow.org/guide/function).
      For more detailed information, see the
      [AutoGraph reference documentation](https://github.com/tensorflow/tensorflow/blob/master/
    tensorflow/python/autograph/g3doc/reference/index.md).

      Args:
        entity: Python callable or class to convert.
        recursive: Whether to recursively convert any functions that the converted
          function may call.
        experimental_optional_features: `None`, a tuple of, or a single
          `tf.autograph.experimental.Feature` value.

      Returns:
        Same as `entity`, the converted Python function or class.

      Raises:
        ValueError: If the entity could not be converted.
    """
    try:
        program_ctx = converter.ProgramContext(
            options=converter.ConversionOptions(
                recursive=recursive,
                user_requested=True,
                optional_features=experimental_optional_features,
            )
        )
        return autograph_artifact(_convert_actual(entity, program_ctx))
    except (ValueError, AttributeError, KeyError, NameError, AssertionError) as e:
        logging.error(1, "Error converting %s", entity, exc_info=True)
        raise ConversionError("converting {}: {}: {}".format(entity, e.__class__.__name__, str(e)))


@export_symbol(v1=["autograph.to_graph"])
def to_graph_v1(
    entity, recursive=True, arg_values=None, arg_types=None, experimental_optional_features=None
):
    """Converts a Python entity into a TensorFlow graph.

    Also see: `tf.autograph.to_code`, `tf.function`.

    Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
    Python code to TensorFlow graph code. It does not implement any caching,
    variable management or create any actual ops, and is best used where greater
    control over the generated TensorFlow graph is desired. Another difference
    from `tf.function` is that `to_graph` will not wrap the graph into a
    TensorFlow function or a Python callable. Internally, `tf.function` uses
    `to_graph`.

    _Example Usage_

    ```python
      def foo(x):
        if x > 0:
          y = x * x
        else:
          y = -x
        return y

      converted_foo = to_graph(foo)

      x = tf.constant(1)
      y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
      assert is_tensor(y)
    ```

    Supported Python entities include:
      * functions
      * classes
      * object methods

    Functions are converted into new functions with converted code.

    Classes are converted by generating a new class whose methods use converted
    code.

    Methods are converted into unbound function that have an additional first
    argument called `self`.

    Args:
      entity: Python callable or class to convert.
      recursive: Whether to recursively convert any functions that the converted
        function may call.
      arg_values: Deprecated.
      arg_types: Deprecated.
      experimental_optional_features: `None`, a tuple of, or a single
        `tf.autograph.experimental.Feature` value.

    Returns:
      Same as `entity`, the converted Python function or class.

    Raises:
      ValueError: If the entity could not be converted.
    """
    del arg_types
    del arg_values
    return to_graph(
        entity, recursive=recursive, experimental_optional_features=experimental_optional_features
    )


@export_symbol(v1=["autograph.to_code"])
def to_code_v1(
    entity,
    recursive=True,
    arg_values=None,
    arg_types=None,
    indentation="  ",
    experimental_optional_features=None,
):
    """Returns the source code generated by AutoGraph, as a string.

    Example usage:

    >>> def f(x):
    ...   if x < 0:
    ...     x = -x
    ...   return x
    >>> tf.autograph.to_code(f)
    "...def tf__f(x):..."

    Also see: `tf.autograph.to_graph`.

    Note: If a function has been decorated with `tf.function`, pass its
    underlying Python function, rather than the callable that `tf.function
    creates:

    >>> @tf.function
    ... def f(x):
    ...   if x < 0:
    ...     x = -x
    ...   return x
    >>> tf.autograph.to_code(f.python_function)
    "...def tf__f(x):..."

    Args:
      entity: Python callable or class.
      recursive: Whether to recursively convert any functions that the converted
        function may call.
      arg_values: Deprecated.
      arg_types: Deprecated.
      indentation: Deprecated.
      experimental_optional_features: `None`, a tuple of, or a single
        `tf.autograph.experimental.Feature` value.

    Returns:
      The converted code as string.
    """
    del arg_values
    del arg_types
    del indentation
    return to_code(
        entity, recursive=recursive, experimental_optional_features=experimental_optional_features
    )


@export_symbol("autograph.to_code", v1=[])
def to_code(entity, recursive=True, experimental_optional_features=None):
    """Returns the source code generated by AutoGraph, as a string.

    Example usage:

    >>> def f(x):
    ...   if x < 0:
    ...     x = -x
    ...   return x
    >>> tf.autograph.to_code(f)
    "...def tf__f(x):..."

    Also see: `tf.autograph.to_graph`.

    Note: If a function has been decorated with `tf.function`, pass its
    underlying Python function, rather than the callable that `tf.function
    creates:

    >>> @tf.function
    ... def f(x):
    ...   if x < 0:
    ...     x = -x
    ...   return x
    >>> tf.autograph.to_code(f.python_function)
    "...def tf__f(x):..."

    Args:
      entity: Python callable or class to convert.
      recursive: Whether to recursively convert any functions that the converted
        function may call.
      experimental_optional_features: `None`, a tuple of, or a single
        `tf.autograph.experimental.Feature` value.

    Returns:
      The converted code as string.
    """
    source = inspect.getsource(
        to_graph(
            entity,
            recursive=recursive,
            experimental_optional_features=experimental_optional_features,
        )
    )
    return textwrap.dedent(source)


_TRANSPILER = None


def initialize_autograph(
    operator_overload=hooks.OperatorBase(),
    converter_name="autograph",
    convert_modules=[],
    do_not_convert_modules=["nvidia.dali._autograph"],
):
    """Initialize the AutoGraph with custom operator overloads.

    Parameters
    ----------
    operator_overload : subclass of autograph.OperatorBase(), optional
        Customization point for detection of user-defined objects that trigger
        the user-defined overload to be called by AutoGraph instead of falling
        back to regular Python semantics, by default autograph.OperatorBase().
    converter_name : str, optional
        Name that is used to generated converted function names and as a fake module under which
        the AutoGraph is inserted into them, by default "autograph".
    convert_modules : list, optional
        Provides a way to include extra modules that should be converted by the autograph.
        In particular, the modules specified here take the precedence over `do_not_convert_modules`,
        so that some submodules of the otherwise excluded modules can be converted.
    do_not_convert_modules : list, optional
        AutoGraph needs to filter the module that should not be converted. By default it will
        only filter out its own functions, provide the list of module that should be ignored.
        If the autograph is used under different name (for example included in the source as
        some_library._ag), this parameter should be adjusted , by default ["autograph"]
    """
    global _TRANSPILER
    if _TRANSPILER is not None:
        raise RuntimeError("AutoGraph already initialized")
    _TRANSPILER = PyToLib(converter_name, operator_overload)
    convert_rules = tuple(config.Convert(name) for name in convert_modules)
    # Add the name of the initialized library to know libraries to stop recursive conversion
    do_not_convert_rules = tuple(config.DoNotConvert(name) for name in do_not_convert_modules)
    config.CONVERSION_RULES = (
        (config.DoNotConvert(converter_name),)
        + convert_rules
        + do_not_convert_rules
        + config.CONVERSION_RULES
    )
