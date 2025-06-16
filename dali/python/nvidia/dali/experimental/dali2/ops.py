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
from . import _tensor, _batch
from . import _eval_context
import nvidia.dali as dali
from typing import Optional


class Operator:
    def __init__(
        self,
        max_batch_size,
        name=None,
        device="cpu",
        num_inputs=None,
        call_arg_names=None,
        **kwargs,
    ):
        self._name = name
        self._max_batch_size = max_batch_size
        self._init_args = kwargs
        self._num_inputs = num_inputs
        self._call_arg_names = None if call_arg_names is None else tuple(call_arg_names)
        if isinstance(device, str):
            self._device = _device.Device(
                name=device,
                device_id=kwargs.get("device_id", _device.Device.default_device_id(device)),
            )
        else:
            if not isinstance(device, _device.Device):
                raise TypeError(
                    f"`device` must be a Device instance or a string, got {type(device)}"
                )
            self._device = device
        self._minipipe = None
        self._input_meta = []
        self._arg_meta = {}
        self._num_outputs = None
        self._output_devices = None
        self._op = None
        self._last_invocation = None

    @classmethod
    def get(
        cls,
        max_batch_size: int,
        name: Optional[str] = None,
        device: Optional[_device.Device] = None,
        num_inputs: Optional[int] = None,
        call_arg_names: Optional[list[str]] = None,
        **init_args,
    ):
        if device is None:
            device = _device.Device.current()
        if not isinstance(device, _device.Device):
            raise TypeError("device must be a Device instance")

        def freeze_arg(arg):
            if isinstance(arg, list):
                return tuple(arg)
            return arg

        def freeze_args(args):
            sorted_keys = sorted(args.keys())
            return tuple([(k, freeze_arg(args[k])) for k in sorted_keys])

        call_arg_names = freeze_arg(call_arg_names)
        key = (device, max_batch_size, num_inputs, call_arg_names, freeze_args(init_args))
        inst = cls._instance_cache.get(key, None)
        if inst is None:
            with device:
                inst = cls(
                    max_batch_size,
                    name=name,
                    device=device,
                    num_inputs=num_inputs,
                    call_arg_names=call_arg_names,
                    **init_args,
                )
                cls._instance_cache[key] = inst
        return inst

    def infer_num_outputs(self, *inputs, **args):
        self._init_pipeline(inputs, args)
        return self._num_outputs

    def infer_output_devices(self, *inputs, **args):
        self._init_pipeline(inputs, args)
        return self._output_devices

    def _init_pipeline(self, inputs, args):
        if self._minipipe is None:
            self._num_inputs = len(inputs)
            self._call_arg_names = tuple(args.keys())
            import nvidia.dali as dali

            self._minipipe = dali.Pipeline(
                batch_size=self._max_batch_size,
                exec_dynamic=True,
                num_threads=4,
                prefetch_queue_depth=1,
                device_id=None if self._device.device_type == "cpu" else self._device.device_id,
            )
            with self._minipipe:
                input_nodes = [
                    dali.fn.external_source(
                        name=f"input_{i}",
                        device=self._device.device_type,
                        no_copy=True,
                        blocking=True,
                    )
                    for i in range(len(inputs))
                ]
                arg_nodes = {name: dali.fn.external_source(name=f"arg_{name}") for name in args}
                op = self.legacy_op(
                    name=self._name, device=self._device.device_type, **self._init_args
                )
                self._op = op
                out = op(*input_nodes, **arg_nodes)
                print(out.source._spec)
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
                if isinstance(out, (tuple, list)):
                    self._minipipe.set_outputs(*out)
                else:
                    self._minipipe.set_outputs(out)
                self._minipipe.build()

            self._set_meta(inputs, args)

    def run(self, *inputs, **args):
        print("Running operator", self._name, type(self), id(self))
        # print("inputs", inputs)
        # print("args", args)
        # print("minipipe", self._minipipe)
        # print("input_meta", self._input_meta)
        # print("arg_meta", self._arg_meta)
        if self._minipipe is not None:
            if self.schema.IsStateful():
                # clearing the _minipipe in a stateful op would destroy the state
                self.check_compatible(inputs, args)
            elif not self.is_compatible(inputs, args):
                # we can reinitialize a stateless operator - not very efficient :(
                self._minipipe = None

        self._init_pipeline(inputs, args)
        for i, input in enumerate(inputs):
            self._minipipe.feed_input(f"input_{i}", self._to_batch(input).evaluate()._backend)
        for name, arg in args.items():
            self._minipipe.feed_input(f"arg_{name}", self._to_batch(arg).evaluate()._backend)
        return self._minipipe.run(_eval_context.EvalContext.get().cuda_stream)

    def _to_batch(self, x):
        if not isinstance(x, _batch.Batch):
            return _batch.Batch([x])
        else:
            return x

    def _set_meta(self, inputs, args):
        self._input_meta = [self._make_meta(input) for input in inputs]
        self._arg_meta = {name: self._make_meta(arg) for name, arg in args.items()}

    def is_compatible(self, inputs, args):
        ret = self._input_meta == [
            self._make_meta(input) for input in inputs
        ] and self._arg_meta == {name: self._make_meta(arg) for name, arg in args.items()}
        return ret

    def check_compatible(self, inputs, args):
        def error_header():
            return f"The invocation of operator {self.display_name} is not compatible with the previous call:\n"

        if len(inputs) != len(self._input_meta):
            raise RuntimeError(
                error_header()
                + f"The number of inputs ({len(inputs)}) does not match the number of inputs used in the previous call ({len(self._input_meta)})"
            )
        for i, input in enumerate(inputs):
            if self._input_meta[i] != self._make_meta(input):
                raise RuntimeError(
                    error_header()
                    + f"The input {i} is not compatible with the input used in the previous call"
                )
        for name, arg in args.items():
            if name not in self._arg_meta:
                raise RuntimeError(
                    error_header() + f"The argument `{name}` was not used in the previous call"
                )
            if self._arg_meta[name] != self._make_meta(arg):
                raise RuntimeError(
                    error_header()
                    + f"The argument `{name}` is not compatible with the argument used in the previous call"
                )
        for name in self._arg_meta:
            if name not in args:
                raise RuntimeError(
                    error_header()
                    + f"The argument `{name}` used in the previous call was not supplied in the current one"
                )

    def _make_meta(self, x):
        is_batch = False
        if isinstance(x, _invocation.Invocation):
            is_batch = x.is_batch
        elif isinstance(x, _batch.Batch):
            is_batch = True
        else:
            is_batch = False

        return {
            "is_batch": is_batch,
            "ndim": x.ndim,
            "layout": x.layout,
            "dtype": x.dtype,
        }

    @property
    def display_name(self):
        if "display_name" in self._init_args:
            type_name = self._init_args["display_name"]
        else:
            type_name = self.schema.OperatorName()
        if self._name is not None:
            return f'type_name "{self._name}"'
        else:
            return type_name


class Reader(Operator):
    def __init__(
        self,
        batch_size=None,
        name=None,
        device="cpu",
        num_inputs=None,
        call_arg_names=None,
        **kwargs,
    ):
        if name is None:
            name = f"Reader_{id(self)}"
        self._actual_batch_size = batch_size
        self._batch_size = batch_size
        super().__init__(
            self._actual_batch_size, name, device, num_inputs, call_arg_names, **kwargs
        )

    def samples(self):
        if not self._minipipe:
            if self._actual_batch_size is None:
                self._actual_batch_size = 1
            if self._max_batch_size is None:
                self._max_batch_size = self._actual_batch_size
            self._init_pipeline((), {})
        meta = self._minipipe.reader_meta(self._name)
        idx = 0
        while idx < meta["epoch_size_padded"]:
            outputs = self.run()
            batch_size = len(outputs[0])
            idx += batch_size
            for x in zip(*outputs):
                yield x

    def batches(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size is None:
            raise ValueError("Batch size was not specified")
        if not self._minipipe:
            if self._max_batch_size and self._max_batch_size < batch_size:
                raise ValueError(
                    f"`batch_size` {batch_size} is larger than the `max_batch_size` {self._max_batch_size} specified when the operator was created"
                )
            self._max_batch_size = batch_size
            self._init_pipeline((), {})
        else:
            if self._max_batch_size and self._max_batch_size != batch_size:
                raise ValueError(
                    f"`batch_size` {batch_size} is different than the `max_batch_size` {self._max_batch_size} used in the previous call"
                )
        meta = self._minipipe.reader_meta(self._name)
        idx = 0
        while idx < meta["epoch_size_padded"]:
            outputs = self.run()
            batch_size = len(outputs[0])
            idx += batch_size
            yield outputs


all_ops = []


def initialize():
    from . import _op_builder

    global all_ops
    all_ops = _op_builder.build_operators()
