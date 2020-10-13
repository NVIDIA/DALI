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
from nvidia.dali.data_node import DataNode as _DataNode
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
        def is_data_node(x):
            return isinstance(x, _DataNode)
        def is_input(x):
            if is_data_node(x):
                return True
            return isinstance(x, (list, tuple)) and \
                   any(isinstance(y, _DataNode) for y in x) and \
                   all(isinstance(y, (_DataNode, nvidia.dali.types.ScalarConstant)) for y in x)
        def is_call_arg(name, value):
            return name == "name" or is_data_node(value)

        def to_scalar(scalar):
            return scalar.value if isinstance(scalar, nvidia.dali.types.ScalarConstant) else scalar

        scalar_args = { name:to_scalar(value) for (name, value) in arguments.items() if not is_call_arg(name, value) }
        tensor_args = { name:value for (name, value) in arguments.items() if is_call_arg(name, value) }
        for idx, inp in enumerate(inputs):
            if not is_input(inp):
                raise TypeError("""Input {0} is neither a DALI `DataNode` nor a tuple of data nodes.
Got {1} instead when calling operator {2}.""".format(idx, type(inp).__name__, op_class.__name__))
        default_dev = nvidia.dali.ops._choose_device(inputs)
        if default_dev == "gpu" and scalar_args.get("device") == "cpu":
            raise ValueError("An operator with device='cpu' cannot accept GPU inputs.")
        if "device" not in scalar_args:
            scalar_args["device"] = default_dev

        return op_class(**scalar_args)(*inputs, **tensor_args)

    op_wrapper.__name__ = wrapper_name
    op_wrapper.__doc__ = "see :class:`{0}.{1}`".format(op_class.__module__, op_class.__name__)
    return op_wrapper

def _wrap_op(op_class, submodule):
    wrapper_name = _to_snake_case(op_class.__name__)
    fn_module = sys.modules[__name__]
    module = _internal.get_submodule(fn_module, submodule)
    if not hasattr(module, wrapper_name):
        wrap_func = _wrap_op_fn(op_class, wrapper_name)
        setattr(module, wrapper_name, wrap_func)
        if submodule:
            wrap_func.__module__ = module.__name__


from nvidia.dali.external_source import external_source
external_source.__module__ = __name__
