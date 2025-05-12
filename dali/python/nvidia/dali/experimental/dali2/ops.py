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
import nvidia.dali as dali

class Operator:
    def __init__(self, max_batch_size, name=None, **kwargs):
        self._name = name
        self._max_batch_size = max_batch_size
        self._init_args = kwargs
        self._device = _device.Device(
            name=kwargs.get("device", "cpu"),
            device_id=kwargs.get("device_id", 0)
        )
        self._minipipe = None
        self._input_meta = []
        self._arg_meta = []

    def run(self, *inputs, **args):
        if self._minipipe is None or not self.is_compatible(inputs, args):
            self._set_meta(inputs, args)
            import nvidia.dali as dali
            self._minipipe = dali.Pipeline(
                 batch_size=self._max_batch_size,
                 exec_dynamic=True,
                 num_threads=4,
                 prefetch_queue_depth=1,
                 device_id=None if self._device.device_type == "cpu" else self._device.device_id)
            with self._minipipe:
                inps = [dali.fn.external_source(name=f"input_{i}", device=self._device.device_type, device_id=self._device.device_id) for i in range(len(inputs))]
                args = [dali.fn.external_source(name=f"arg_{name}", device=self._device.device_type, device_id=self._device.device_id) for name in args]
                op = self.legacy_op(max_batch_size=self._max_batch_size, name=self._name, **self._init_args)
                out = op(*inps, **args)
                self._minipipe.set_outputs(out)
                self._minipipe.build()
        return self._minipipe.run()

    def _set_meta(self, inputs, args):
        self._input_meta = [self._make_meta(input) for input in inputs]
        self._arg_meta = [self._make_meta(arg) for arg in args]

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




def initialize():
    from . import _op_builder
    _op_builder.build_operators()
