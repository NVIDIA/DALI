# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import _device
from . import _invocation
from . import _tensor, _tensor_list
from . import _eval_context
import nvidia.dali as dali

class Operator:
    def __init__(self, max_batch_size, name=None, **kwargs):
        self._name = name
        self._max_batch_size = max_batch_size
        self._init_args = kwargs
        device_type = kwargs.get("device", "cpu")
        self._device = _device.Device(
            name=device_type,
            device_id=kwargs.get("device_id", _device.Device.default_device_id(device_type))
        )
        self._minipipe = None
        self._input_meta = []
        self._arg_meta = {}
        self._num_outputs = None
        self._output_devices = None
        self._op = None

    def infer_num_outputs(self, *inputs, **args):
        self._init_pipeline(inputs, args)
        return self._num_outputs

    def infer_output_devices(self, *inputs, **args):
        self._init_pipeline(inputs, args)
        return self._output_devices

    def _init_pipeline(self, inputs, args):
        if self._minipipe is None:
            print("init pipeline")
            print(inputs)
            print(args)
            import nvidia.dali as dali
            self._minipipe = dali.Pipeline(
                 batch_size=self._max_batch_size,
                 exec_dynamic=True,
                 num_threads=4,
                 prefetch_queue_depth=1,
                 device_id=None if self._device.device_type == "cpu" else self._device.device_id)
            with self._minipipe:
                input_nodes = [dali.fn.external_source(name=f"input_{i}", device=self._device.device_type) for i in range(len(inputs))]
                arg_nodes = { name: dali.fn.external_source(name=f"arg_{name}") for name in args }
                op = self.legacy_op(name=self._name, **self._init_args)
                self._op = op
                out = op(*input_nodes, **arg_nodes)
                if isinstance(out, (tuple, list)):
                    self._output_devices = []
                    self._num_outputs = len(out)
                    for o in out:
                        device_type = o.device
                        device_id = self._device.device_id
                        self._output_devices.append(_device.Device(device_type, device_id))
                else:
                    self._num_outputs = 1
                    self._output_devices = [_device.Device(out.device, self._device.device_id)]
                self._minipipe.set_outputs(out)
                self._minipipe.build()

    def run(self, *inputs, **args):
        if self._minipipe is not None and self._input_meta and self._arg_meta and not self.is_compatible(inputs, args):
            self._minipipe = None

        self._init_pipeline(inputs, args)
        self._set_meta(inputs, args)
        for i, input in enumerate(inputs):
            self._minipipe.feed_input(f"input_{i}", input.evaluate()._backend)
        for name, arg in args.items():
            self._minipipe.feed_input(f"arg_{name}", arg.evaluate()._backend)
        return self._minipipe.run(_eval_context.EvalContext.get().cuda_stream)

    def _set_meta(self, inputs, args):
        print("set_meta",inputs, args)
        self._input_meta = [self._make_meta(input) for input in inputs]
        self._arg_meta = { name: self._make_meta(arg) for name, arg in args.items() }

    def is_compatible(self, inputs, args):
        return self._input_meta == [input.meta for input in inputs] and self._arg_meta == [arg.meta for arg in args]

    def _make_meta(self, x):
        is_batch = False
        if isinstance(x, _invocation.Invocation):
            is_batch = x.is_batch
        elif isinstance(x, _tensor_list.TensorList):
            is_batch = True
        else:
            is_batch = False

        return {
            "is_batch": is_batch,
            "ndim": x.ndim,
            "layout": x.layout,
            "dtype": x.dtype,
        }


all_ops = []

def initialize():
    from . import _op_builder
    global all_ops
    all_ops = _op_builder.build_operators()
