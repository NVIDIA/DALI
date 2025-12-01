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
from . import _eval_context
import nvidia.dali as dali
from typing import Optional
import nvidia.dali.backend_impl as _b
from ._tensor import Tensor
from ._batch import Batch


class Operator:
    def __init__(
        self,
        max_batch_size,
        name=None,
        device="cpu",
        **kwargs,
    ):
        """Constructs an operator instance.
        Parameters
        ----------
        max_batch_size : int
            The maximum batch size for this operator instance.
        name : str, optional
            The name of the operator instance.
        device : Device or str, optional
            The device where the operation is executed.
        """
        self._name = name
        self._max_batch_size = max_batch_size
        self._init_args = kwargs
        self._api_type = None

        from ._device import device as _to_device

        self._device = _to_device(device)

        self._input_meta = []
        self._arg_meta = {}
        self._num_outputs = None
        self._output_devices = None
        self._op_inst = None
        self._op_backend = None
        self._op_spec = None
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
        """Gets an operator instance for a specified set of parameters."""
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
                    **init_args,
                )
                cls._instance_cache[key] = inst
        return inst

    def infer_num_outputs(self, *inputs, **args):
        self._init_spec(inputs, args)
        return self._num_outputs

    def input_device(self, index: int, actual_device: Optional[str] = None):
        default_input_device = "gpu" if self._device.device_type == "gpu" else "cpu"
        dev_type = self.schema.GetInputDevice(index, actual_device, default_input_device)
        if dev_type is None:
            return self._device
        return _device.Device(dev_type, self._device.device_id)  # inherit the device id

    def infer_output_devices(self, *inputs, **args):
        self._init_spec(inputs, args)
        return self._output_devices

    def _pre_call(self, *inputs, **args):
        pass

    def _is_backend_initialized(self):
        return self._op_backend is not None

    def _reset_backend(self):
        self._op_backend = None
        self._op_spec = None

    def _init_spec(self, inputs, args):
        if self._op_spec is None:
            import nvidia.dali as dali

            with self._device:
                # Create fake DataNodes (they're quite lightweight) for the inputs and arguments,
                # so we can use the ops API to obtain an OpSpec.
                input_nodes = [
                    dali.data_node.DataNode(
                        name=f"input_{i}", device=inputs[i].device.device_type, source=None
                    )
                    for i in range(len(inputs))
                ]
                arg_nodes = {
                    name: dali.data_node.DataNode(name=f"arg_{name}", device="cpu", source=None)
                    for name in args
                }

                # legacy_op is a member of the old `ops` module - we use the ops API to obtain
                # an OpSpec
                op = self.legacy_op(
                    name=self._name, device=self._device.device_type, **self._init_args
                )
                self._op_inst = op
                out = op(*input_nodes, **arg_nodes)
                if isinstance(out, (list, tuple)):
                    spec = out[0].source.spec
                else:
                    spec = out.source.spec

                self._op_spec = spec

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

                self._set_meta(inputs, args)

    def _init_backend(self, ctx, inputs, args):
        if self._op_backend is not None:
            return

        if ctx is None:
            ctx = _eval_context.EvalContext.current()
        with self._device:
            with ctx:
                self._init_spec(inputs, args)
                if ctx._thread_pool is not None:
                    self._op_spec.AddArg("num_threads", ctx._thread_pool.num_threads)
                else:
                    self._op_spec.AddArg("num_threads", 1)
                self._op_spec.AddArg(
                    "device_id",
                    (
                        self._device.device_id
                        if self._device.device_type == "gpu" or self._device.device_type == "mixed"
                        else dali.types.CPU_ONLY_DEVICE_ID
                    ),
                )
                if self._max_batch_size is None:
                    self._max_batch_size = 1
                self._op_spec.AddArg("max_batch_size", self._max_batch_size)
                self._op_backend = _b._Operator(self._op_spec)

    def run(self, ctx, *inputs, batch_size=None, **args):
        device_id = ctx.device_id if ctx is not None else None
        device_ctx = (
            _device.Device("gpu", device_id) if device_id is not None else _device.Device("cpu")
        )
        with device_ctx:
            if (
                batch_size is not None
                and self._max_batch_size is not None
                and batch_size > self._max_batch_size
                and self.schema.IsStateful()
            ):
                raise RuntimeError(
                    f"The batch size {batch_size} is larger than the `max_batch_size` "
                    f"{self._max_batch_size} specified when the operator was created."
                )

            def _is_batch():
                for input in inputs:
                    if isinstance(input, ((_b.TensorListCPU, _b.TensorListGPU))):
                        return True
                for input in args.values():
                    if isinstance(input, ((_b.TensorListCPU, _b.TensorListGPU))):
                        return True
                return False

            is_batch = batch_size is not None or _is_batch()
            if self._is_backend_initialized():
                if self.schema.IsStateful():
                    # clearing the backend in a stateful op would destroy the state
                    self.check_compatible(inputs, batch_size, args)
                elif not self.is_compatible(inputs, batch_size, args):
                    # we can reinitialize a stateless operator - not very efficient :(
                    self._reset_backend()

            self._init_backend(ctx, inputs, args)
            workspace = _b._Workspace(ctx._thread_pool, ctx._cuda_stream)
            for i, input in enumerate(inputs):
                workspace.AddInput(self._to_batch(input).evaluate()._storage)
            for name, arg in args.items():
                workspace.AddArgumentInput(name, self._to_batch(arg).evaluate()._storage)
            self._op_backend.SetupAndRun(workspace, batch_size)
            out = workspace.GetOutputs()
            if is_batch:
                return tuple(out)
            else:
                tensors = tuple(o[0] for o in out)
                return tensors

    def _to_batch(self, x):
        if not isinstance(x, Batch):
            return Batch([x])
        else:
            return x

    def _set_meta(self, inputs, args):
        self._input_meta = [self._make_meta(input) for input in inputs]
        self._arg_meta = {name: self._make_meta(arg) for name, arg in args.items()}

    def is_compatible(self, inputs, batch_size, args):
        """Checks if the inputs and arguments are compatible with this operator instance."""
        if batch_size is not None:
            if batch_size > self._max_batch_size:
                return False
        if self._input_meta != [self._make_meta(input) for input in inputs]:
            return False
        if self._arg_meta != {name: self._make_meta(arg) for name, arg in args.items()}:
            return False
        return True

    def check_compatible(self, inputs, batch_size, args):
        """Raises an error if the inputs and arguments are not compatible with this op instance."""

        def error_header():
            return (
                f"The invocation of operator {self.display_name} "
                f"is not compatible with the previous call:\n"
            )

        if batch_size is not None:
            if batch_size > self._max_batch_size:
                raise RuntimeError(
                    f"{error_header()}"
                    f"The batch size {batch_size} is larger than the `max_batch_size` "
                    f"{self._max_batch_size} specified when the operator was created."
                )

        if len(inputs) != len(self._input_meta):
            raise RuntimeError(
                f"{error_header()}"
                f"The number of inputs ({len(inputs)}) does not match the number "
                f"of inputs used in the previous call ({len(self._input_meta)})."
            )
        for i, input in enumerate(inputs):
            if self._input_meta[i] != self._make_meta(input):
                raise RuntimeError(
                    f"{error_header()}"
                    f"The input {i} is not compatible with the input used in the previous call."
                )
        for name, arg in args.items():
            if name not in self._arg_meta:
                raise RuntimeError(
                    f"{error_header()}" f"The argument `{name}` was not used in the previous call."
                )
            if self._arg_meta[name] != self._make_meta(arg):
                raise RuntimeError(
                    f"{error_header()}"
                    f"The argument `{name}` is not compatible with the argument used in the "
                    f"previous call."
                )
        for name in self._arg_meta:
            if name not in args:
                raise RuntimeError(
                    f"{error_header()}"
                    f"The argument `{name}` used in the previous call was not supplied in the "
                    f"current one."
                )

    def _make_meta(self, x):
        is_batch = False
        if isinstance(x, _invocation.Invocation):
            is_batch = x.is_batch
        elif isinstance(x, Batch):
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
        **kwargs,
    ):
        if name is None:
            name = f"Reader_{id(self)}"
        self._actual_batch_size = batch_size
        self._batch_size = batch_size
        super().__init__(self._actual_batch_size, name, device, **kwargs)

    def _pre_call(self, *inputs, **args):
        if self._api_type is None:
            self._api_type = "run"
        elif self._api_type != "run":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `run`/`__call__` on the same reader."
            )

    def run(self, ctx=None, *inputs, **args):
        """
        Runs the reader and obtains one result (batch or sample, depending on `batch_size`).

        Do not call this function directly. Use `__call__` instead.
        """
        if self._api_type is None:
            self._api_type = "run"
        elif self._api_type != "run":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `run`/`__call__` on the same reader."
            )

        return super().run(ctx, *inputs, **args)

    def next_epoch(self, batch_size=None, ctx: Optional[_eval_context.EvalContext] = None):
        """
        Obtains an iterator that goes over the next epoch from the reader.

        The return value is an iterator that returns either individual samples (if `batch_size` is
        ``None`` and was not specified at construction) or batches (if `batch_size` was specified
        here or at construction).

        This iterator will go over the dataset (or shard, if sharding was specified at construction)
        once.

        .. note::
            The iterator must be traversed completely before the next call to `next_epoch` is made.
            Therefore, it is impossible to traverse one reader using two iterators.
            If another iterator is necessary, create a separate reader instance.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size is not None:
            return self._batches(batch_size, ctx)
        else:
            return self._samples(ctx)

    def _samples(self, ctx: Optional[_eval_context.EvalContext] = None):
        if self._api_type is None:
            self._api_type = "samples"
        elif self._api_type != "samples":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `run`/`__call__` on the same reader."
            )

        if ctx is None:
            ctx = _eval_context.EvalContext.current()
        with ctx:
            if not self._is_backend_initialized():
                if self._actual_batch_size is None:
                    self._actual_batch_size = 1
                if self._max_batch_size is None:
                    self._max_batch_size = self._actual_batch_size
                self._init_backend(ctx, (), {})
            meta = self._op_backend.GetReaderMeta()
            idx = 0
            while idx < meta["epoch_size_padded"]:
                outputs = super().run(ctx, batch_size=self._actual_batch_size)
                batch_size = len(outputs[0])
                assert batch_size == self._actual_batch_size
                idx += batch_size
                for x in zip(*outputs):
                    outs = tuple(Tensor(o) for o in x)
                    yield outs

    def _batches(self, batch_size=None, ctx: Optional[_eval_context.EvalContext] = None):
        if self._api_type is None:
            self._api_type = "batches"
        elif self._api_type != "batches":
            raise RuntimeError("Cannot mix samples(), batches() and run() on the same reader.")

        if ctx is None:
            ctx = _eval_context.EvalContext.current()
        with ctx:
            if batch_size is None:
                batch_size = self._batch_size
            if batch_size is None:
                raise ValueError("Batch size was not specified")
            if not self._op_backend:
                if self._max_batch_size and self._max_batch_size < batch_size:
                    raise ValueError(
                        f"`batch_size` {batch_size} is larger than the `max_batch_size` "
                        f"{self._max_batch_size} specified when the operator was created"
                    )
                self._max_batch_size = batch_size
                self._init_backend(ctx, (), {})
            else:
                if self._max_batch_size and self._max_batch_size != batch_size:
                    raise ValueError(
                        f"`batch_size` {batch_size} is different than the `max_batch_size` "
                        f"{self._max_batch_size} used in the previous call"
                    )
            meta = self._op_backend.GetReaderMeta()
            idx = 0
            while idx < meta["epoch_size_padded"]:
                outputs = super().run(ctx, batch_size=batch_size)
                batch_size_returned = len(outputs[0])
                assert batch_size_returned == batch_size
                idx += batch_size_returned
                yield tuple(Batch(o) for o in outputs)


_all_ops = []
_all_functions = []


def _initialize():
    from . import _op_builder

    global _all_ops, _all_functions
    _all_ops, _all_functions = _op_builder.build_operators()
