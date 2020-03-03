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
from __future__ import division
import sys
from nvidia.dali.data_node import DataNode as _DataNode

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
        else:
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

    if nupper > 0:
        out += pascal[start:i].lower()
    out = _handle_special_case(out)
    return out

def _wrap_op_fn(op_class, wrapper_name):
    def op_wrapper(*inputs, **arguments):
        import nvidia.dali.ops
        def is_data_node(x):
            return isinstance(x, _DataNode)
        def is_input(x):
            return is_data_node(x) or all(isinstance(y, _DataNode) for y in x)
        def is_call_arg(name, value):
            return name == "name" or is_data_node(value)

        scalar_args = { name:value for (name, value) in arguments.items() if not is_call_arg(name, value) }
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
    return op_wrapper

def _wrap_op(op_class):
    wrapper_name = _to_snake_case(op_class.__name__)
    if not hasattr(sys.modules[__name__], wrapper_name):
        setattr(sys.modules[__name__], wrapper_name, _wrap_op_fn(op_class, wrapper_name))

from nvidia.dali.external_source import external_source
external_source.__module__ = __name__
