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
import nvidia.dali.ops
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

def _wrap_op_fn(op_class):
    def op_wrapper(*args, **kwargs):
        def is_edge(x):
            return isinstance(x, _DataNode)
        def is_call_arg(name, value):
            return name == "name" or is_edge(value)

        scalar_args = { name:value for (name, value) in kwargs.items() if not is_call_arg(name, value) }
        tensor_args = { name:value for (name, value) in kwargs.items() if is_call_arg(name, value) }
        for idx, inp in enumerate(args):
            if not is_edge(inp):
                raise TypeError("""Input {0} is not a DALI tensor (edge reference).
Got {1} instead when calling operator {2}.""".format(idx, type(inp).__name__, op_class.__name__))
        default_dev = nvidia.dali.ops._choose_device(args)
        if default_dev == "gpu" and scalar_args.get("device") == "cpu":
            raise ValueError("An operator with device='cpu' cannot accept GPU inputs.")
        if "device" not in scalar_args:
            scalar_args["device"] = default_dev

        return op_class(**scalar_args)(*args, **tensor_args)
    return op_wrapper

def _wrap_op(op_class):
    wrapper_name = _to_snake_case(op_class.__name__)
    if not hasattr(sys.modules[__name__], wrapper_name):
        setattr(sys.modules[__name__], wrapper_name, _wrap_op_fn(op_class))


def external_source(source = None, num_outputs = None, *, cycle = False, name = None, device = "cpu", layout = None):
    """
    Creates a data node which is populated with data from a Python source.
    The data can be provided by the `source` function or iterable, or it can be provided by
    `pipeline.feed_input(name, data, layout)` inside `pipeline.iter_setup`.

    `source` : callable or iterable
    The source of the data. The source is polled for data (via a call or `next(source)` whenever
    the pipeline needs input for the next iteration. The source can supply one or data batches,
    depending on the value or `num_outputs`. If `num_outputs` is not set, the `source` is expected
    to return a single batch. If it's specified, the data is expected to a tuple or list where each
    element corresponds to respective return value of the external_source.

    `cycle`: bool
    If `True`, the source iterable will be wrapped. Otherwise, StopIteration error wil be raised
    when end of data is reached. Setting this flag to True when `source` is not an iterable is an
    error.

    `num_outputs` : int, optional
    If specified, denotes the number of TensorLists produced by the source function

    `name` : str, optional
    The name of the data node - used when feeding the data in `iter_setup`; can be omitted if
    the data is provided by `source`.

    `layout` : str or list/tuple of str:
    If provided, sets the layout of the data. When `num_outputs` > 1, layout can be a list
    containing a distinct layout for each output. If the list has fewer elements than `num_outputs`,
    only the first outputs have the layout set, the reset have it cleared.
    """
    if num_outputs is not None:
        if source is None:
            raise ValueError("The parameter `num_outputs` is only valid when using `source` to "
                "provide data. To feed multiple external sources in `feed_input`, use multiple "
                "`external_source` nodes.")

    op = nvidia.dali.ops.ExternalSource(device = device, num_outputs = num_outputs,
                                        source = source, cycle = cycle, layout = layout)
    return op(name = name)
