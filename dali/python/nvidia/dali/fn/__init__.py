# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=no-member
import sys
from nvidia.dali import backend as _b
from nvidia.dali import internal as _internal
from nvidia.dali.external_source import external_source
from nvidia.dali._utils import dali_trace as _dali_trace

_special_case_mapping = {"b_box": "bbox", "mx_net": "mxnet", "tf_record": "tfrecord"}


def _handle_special_case(s):
    for artifact, desired in _special_case_mapping.items():
        s = s.replace(artifact, desired)
    return s


def _to_snake_case(pascal):
    out = ""
    nupper = 0
    start = 0
    for i, c in enumerate(pascal):
        if c.isupper():
            if nupper == 0:
                start = i
            nupper += 1
        elif c.islower():
            if nupper == 0:
                out += c
            else:
                # do not add another leading underscore
                if len(out) > 0 and out[-1] != "_":
                    out += "_"
                if nupper > 1:
                    out += pascal[start : i - 1].lower() + "_"
                out += pascal[i - 1].lower()
                out += c
                nupper = 0
            start = i + 1
        else:
            out += pascal[start : i + 1].lower()
            start = i + 1
            nupper = 0

    if nupper > 0:
        if len(out) and out[-1].islower():
            out += "_"
        out += pascal[start:].lower()
    out = _handle_special_case(out)
    return out


def _wrap_op_fn(op_class, wrapper_name, wrapper_doc):
    def op_wrapper(*inputs, **kwargs):
        import nvidia.dali.ops

        init_args, call_args = nvidia.dali.ops._separate_kwargs(kwargs)

        default_dev = nvidia.dali.ops._choose_device(inputs)

        if "device" not in init_args:
            init_args["device"] = default_dev

        # TODO(klecki): It is now possible to handle creation of _OperatorInstance directly
        return op_class(**init_args)(*inputs, **call_args)

    def fn_wrapper(*inputs, **kwargs):
        from nvidia.dali._debug_mode import _PipelineDebug

        # This is executed after the APIs are populated and the module is updated
        kwargs.update(
            {
                "_module": fn_wrapper.__module__.replace(".hidden", ""),
                "_display_name": wrapper_name,
            }
        )

        current_pipeline = _PipelineDebug.current()
        if getattr(current_pipeline, "_debug_on", False):
            return current_pipeline._wrap_op_call(op_class, wrapper_name, *inputs, **kwargs)
        else:
            if _dali_trace.is_tracing_enabled():
                kwargs = {**kwargs, "_definition_frame_end": _dali_trace.get_stack_depth() - 1}
            return op_wrapper(*inputs, **kwargs)

    from nvidia.dali.ops import _signatures
    from nvidia.dali import backend as _b

    fn_wrapper.__name__ = wrapper_name
    fn_wrapper.__qualname__ = wrapper_name
    fn_wrapper.__doc__ = wrapper_doc
    schema = _b.TryGetSchema(op_class.schema_name)
    if schema is not None:
        fn_wrapper.__signature__ = _signatures._call_signature(
            schema, include_inputs=True, include_kwargs=True, filter_annotations=True
        )
    fn_wrapper._schema_name = op_class.schema_name
    fn_wrapper._generated = getattr(op_class, "_generated", False)
    return fn_wrapper


def _wrap_op(op_class, submodule, parent_module, wrapper_doc):
    """Wrap the DALI Operator with fn API and insert the function into appropriate module.

    Args:
        op_class: Op class to wrap
        submodule: Additional submodule (scope)
        parent_module (str): If set to None, the wrapper is placed in nvidia.dali.fn module,
            otherwise in a specified parent module.
        wrapper_doc (str): Documentation of the wrapper function
    """
    from nvidia.dali._utils import eager_utils

    schema = _b.TryGetSchema(op_class.schema_name)
    make_hidden = schema.IsDocHidden() if schema else False
    wrapper_name = _to_snake_case(op_class.__name__)

    # Add operator to eager API.
    eager_utils._wrap_eager_op(
        op_class, submodule, parent_module, wrapper_name, wrapper_doc, make_hidden
    )

    if parent_module is None:
        fn_module = sys.modules[__name__]
    else:
        fn_module = sys.modules[parent_module]
    module = _internal.get_submodule(fn_module, submodule)
    if not getattr(op_class, "_generated", False):
        impl_module_override = op_class._impl_module
    else:
        impl_module_override = None
    if not hasattr(module, wrapper_name):
        wrap_func = _wrap_op_fn(op_class, wrapper_name, wrapper_doc)
        setattr(module, wrapper_name, wrap_func)
        _internal._adjust_operator_module(wrap_func, fn_module, submodule, impl_module_override)
        if make_hidden:
            parent_module = _internal.get_submodule(fn_module, submodule[:-1])
            setattr(parent_module, wrapper_name, wrap_func)


external_source._schema_name = "ExternalSource"
_internal._adjust_operator_module(external_source, sys.modules[__name__], [])
