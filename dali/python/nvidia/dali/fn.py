# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali import backend as _b
from nvidia.dali import internal as _internal

_special_case_mapping = {
    "b_box" : "bbox",
    "mx_net" : "mxnet",
    "tf_record" : "tfrecord"
}

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
                if len(out) > 0:
                    out += '_'
                if nupper > 1:
                    out += pascal[start:i-1].lower() + '_'
                out += pascal[i-1].lower()
                out += c
                nupper = 0
            start = i+1
        else:
            out += pascal[start:i+1].lower()
            start = i + 1
            nupper = 0

    if nupper > 0:
        if len(out) and out[-1].islower():
            out += '_'
        out += pascal[start:].lower()
    out = _handle_special_case(out)
    return out

def _wrap_op_fn(op_class, wrapper_name):
    def op_wrapper(*inputs, **arguments):
        import nvidia.dali.ops
        init_args, call_args = nvidia.dali.ops._separate_kwargs(arguments)

        default_dev = nvidia.dali.ops._choose_device(inputs)
        if default_dev == "gpu" and init_args.get("device") == "cpu":
            raise ValueError("An operator with device='cpu' cannot accept GPU inputs.")

        if "device" not in init_args:
            init_args["device"] = default_dev

        return op_class(**init_args)(*inputs, **call_args)

    op_wrapper.__name__ = wrapper_name
    op_wrapper.__doc__ = "see :class:`{0}.{1}`".format(op_class.__module__, op_class.__name__)
    return op_wrapper

def _wrap_op(op_class, submodule):
    schema = _b.TryGetSchema(op_class.__name__)
    make_hidden = schema.IsDocHidden() if schema else False
    wrapper_name = _to_snake_case(op_class.__name__)
    fn_module = sys.modules[__name__]
    module = _internal.get_submodule(fn_module, submodule)
    if not hasattr(module, wrapper_name):
        wrap_func = _wrap_op_fn(op_class, wrapper_name)
        setattr(module, wrapper_name, wrap_func)
        if submodule:
            wrap_func.__module__ = module.__name__
        if make_hidden:
            parent_module = _internal.get_submodule(fn_module, submodule[:-1])
            setattr(parent_module, wrapper_name, wrap_func)


from nvidia.dali.external_source import external_source
external_source.__module__ = __name__
