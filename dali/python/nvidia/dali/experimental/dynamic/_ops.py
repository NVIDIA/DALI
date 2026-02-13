# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.backend_impl as _b
import math
from . import _eval_context, _invocation, _device, _type
from ._batch import Batch, as_batch as _as_batch, _get_batch_size
from ._device import Device, DeviceLike
from ._tensor import Tensor

import nvtx


def _to_tensor(x, device=None, dtype=None):
    with nvtx.annotate("to_tensor", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Tensor):
            if dtype is not None and x.dtype != dtype:
                return Tensor(x, dtype=dtype, device=device)
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                raise ValueError("Batch invocation result cannot be used as a single tensor")
            return Tensor(invocation_result=x, device=device)
        return Tensor(x, device=device, dtype=dtype)


def _to_batch(x, batch_size, device=None, dtype=None):
    with nvtx.annotate("to_batch", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Batch):
            if dtype is not None and x.dtype != dtype:
                return _as_batch(x, dtype=dtype, device=device)
            if device is not None:
                return x.to_device(device)
            return x
        if isinstance(x, _invocation.InvocationResult):
            if x.is_batch:
                return Batch(invocation_result=x, device=device, dtype=dtype)
            else:
                x = _to_tensor(x, dtype=dtype)  # fall back to regular replication
        actual_batch_size = _get_batch_size(x)
        if actual_batch_size is not None:
            if batch_size is not None and actual_batch_size != batch_size:
                raise ValueError(f"Unexpected batch size: {actual_batch_size} != {batch_size}")
            return Batch(x, device=device, dtype=dtype)

        return Batch.broadcast(x, batch_size, device=device, dtype=dtype)


def _get_input_device(x):
    with nvtx.annotate("get_input_device", domain="op_builder"):
        if x is None:
            return None
        if isinstance(x, Batch):
            return x.device
        if isinstance(x, Tensor):
            return x.device
        if isinstance(x, _b.TensorListCPU):
            return _device.Device("cpu")
        if isinstance(x, _b.TensorListGPU):
            return _device.Device("gpu")
        if hasattr(x, "__cuda_array_interface__"):
            return _device.Device("gpu")
        if hasattr(x, "__dlpack_device__"):
            dev = x.__dlpack_device__()
            if int(dev[0]) == 1 or int(dev[0]) == 3:  # CPU or CPU_PINNED
                return _device.Device("cpu")
            elif int(dev[0]) == 2:
                return _device.Device("gpu", dev[1])
            else:
                raise ValueError(f"Unknown DLPack device type: {dev.type}")
        if hasattr(x, "__dlpack__"):
            return _device.Device("cpu")
        if isinstance(x, list) and x:
            return _get_input_device(x[0])
        return None


def _infer_batch_size(explicit_batch_size, *raw_args, **raw_kwargs):
    if explicit_batch_size is not None:
        return explicit_batch_size
    batch_size = None
    with nvtx.annotate("_infer_batch_size", domain="op_builder"):
        for i, x in enumerate(list(raw_args) + list(raw_kwargs.values())):
            x_batch_size = _get_batch_size(x)
            if x_batch_size is not None:
                if batch_size is not None:
                    if x_batch_size != batch_size:
                        raise ValueError(f"Inconsistent batch size: {x_batch_size} != {batch_size}")
                else:
                    batch_size = x_batch_size
    return batch_size


class Operator:
    """Base class for all dynamic operators. Manages backend lifecycle, caching, and execution.

    The actual operator subclasses are constructed via _op_builder.build_operator_class() factory
    function.

    Operator._get() can be used instead of the constructor to utilize the instance caching.
    """

    # Class members - each subclass will override in the factory function:
    _schema = None
    _schema_name = None
    _supported_backends = frozenset()
    _op_name = None  # CamelCase legacy class API name, without the module - e.g. "CoinFlip"
    _fn_name = None  # snake_case api name - e.g. "coin_flip"
    _legacy_op = None  # The legacy operator class from the nvidia.dali.ops module
    _is_stateful = False
    _has_random_state_arg = False
    # Indicates if this operator is generated and we can autogenerate the stubs or we need
    # to reimport the operator from py to pyi file.
    _generated = False

    def __init__(
        self,
        max_batch_size,
        name=None,
        device="cpu",
        *,
        _backend=None,
        **kwargs,
    ):
        """Constructs an operator instance
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

        self._device = _device.device(device)
        if _backend is None:
            if self._device.device_type in self._supported_backends:
                _backend = self._device.device_type
            elif self._device.device_type == "gpu" and "mixed" in self._supported_backends:
                _backend = "mixed"
            else:
                raise ValueError(f'Invalid device "{device}" for operator `{self._schema_name}`')
        else:
            # _backend is an internal parameter - once it's passed explicitly, it must be correct
            assert _backend in self._supported_backends, "Internal error: incompatible backend."
        self._backend = _backend

        # Information below is lazy-initialized
        # TODO(klecki): Use @property or @cached_property for self-init.

        # Metadata about batch/sample, layout, dim and type. See _make_meta() for more details.
        self._input_meta = []
        self._arg_meta = {}
        # Number of outputs
        self._num_outputs = None
        # When an operator (e.g. TFRecord) returns a dictionary, outputs are named
        self._output_names = None
        # Expected device placement of the outputs
        self._output_devices = None
        # Instance of the legacy Python Operator from the nvidia.dali.ops module
        self._op_inst = None
        # Instance of the C++ OperatorBase class - used for direct invocation of operator
        self._op_backend = None
        self._op_spec = None
        self._last_invocation = None

    @classmethod
    def _get(
        cls,
        max_batch_size: int,
        name: str | None = None,
        device: DeviceLike | None = None,
        num_inputs: int | None = None,
        call_arg_names: list[str] | None = None,
        _backend: str | None = None,
        *,
        inputs,
        init_args,
        call_args,
    ):
        """Gets an operator instance for a specified set of parameters."""
        if device is None:
            device = Device.current()
        if not isinstance(device, Device):
            raise TypeError("device must be a Device instance")

        def freeze_arg(arg):
            if isinstance(arg, list):
                return tuple(arg)
            return arg

        def freeze_args(args):
            sorted_keys = sorted(args.keys())
            return tuple([(k, freeze_arg(args[k])) for k in sorted_keys])

        call_arg_names = freeze_arg(call_arg_names)
        key = (
            cls,
            _backend,
            device,
            max_batch_size,
            num_inputs,
            call_arg_names,
            freeze_args(init_args),
            tuple((cls._make_meta(input) for input in inputs)),
            freeze_args({name: cls._make_meta(arg) for name, arg in call_args.items()}),
        )
        ctx = _eval_context.EvalContext.current()
        inst = ctx._instance_cache.pop(key, None)
        if inst is None:
            with device:
                inst = cls(
                    max_batch_size,
                    name=name,
                    device=device,
                    _backend=_backend,
                    **init_args,
                )
        inst._cache = ctx._instance_cache
        inst._key = key
        return inst

    def _infer_num_outputs(self, *inputs, **args):
        self._init_spec(inputs, args)
        return self._num_outputs

    @classmethod
    def _input_device(
        cls,
        backend: str,
        index: int,
        actual_device: Device | None = None,
        operator_device: Device | None = None,
    ):
        default_input_device = "gpu" if backend == "gpu" else "cpu"
        actual_device_type = actual_device.device_type if actual_device is not None else None
        dev_type = cls._schema.GetInputDevice(index, actual_device_type, default_input_device)
        if dev_type is None:
            return operator_device
        if dev_type == "cpu":
            dev_id = None
        else:
            if backend != "cpu":
                dev_id = operator_device.device_id  # we need to match our current device
            else:
                # This is a CPU operator so it doesn't have a device id - we should just
                # use whatever was passed in.
                dev_id = actual_device.device_id if actual_device is not None else None

        return Device(dev_type, dev_id)  # inherit the device id

    @classmethod
    def _process_params(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs):
        """
        Processes run-time parmaeters passed to the operator to ones that can be consumed DALI
        (Batch or Tensor).

        This is a class method, as it doesn't require an operator instance - and this method
        is essential for proper operator instance caching, as input/argument metadata is a part
        of the operator cache key.
        """
        is_batch = batch_size is not None
        if cls._has_random_state_arg:
            from . import random

            rng = raw_kwargs.pop("rng", None)
            if rng is None:
                rng = random.get_default_rng()
            if not isinstance(rng, random.RNG):
                raise ValueError(
                    f"rng must be an instance of nvidia.dali.experimental.dynamic.random.RNG, "
                    f"but got {type(rng)}"
                )

            # Use the provided RNG to generate 7 random uint32 values.
            # This creates a fixed-size random state tensor.
            # 7 uint32 words = 224 bits; required is 194 bits (operator reads first 25 bytes).
            # Only one random state tensor is created per call, not per sample.
            raw_kwargs["_random_state"] = Tensor(
                [rng() for _ in range(7)],
                dtype=_type.uint32,
                device="cpu",
            )

        inputs = []
        kwargs = {}

        if is_batch:
            with nvtx.annotate("__call__: convert to batches", domain="op_builder"):
                for i, inp in enumerate(raw_args):
                    if inp is None:
                        continue
                    input_device = cls._input_device(backend, i, _get_input_device(inp), op_device)
                    inp = _to_batch(inp, batch_size, device=input_device)
                    inputs.append(inp)
                for k, v in raw_kwargs.items():
                    if v is None:
                        continue
                    dtype = cls._argument_conversion_map[k]
                    kwargs[k] = _to_batch(v, batch_size, device=_device.Device("cpu"), dtype=dtype)
        else:
            with nvtx.annotate("__call__: convert to tensors", domain="op_builder"):
                for inp in raw_args:
                    if inp is None:
                        continue
                    inputs.append(_to_tensor(inp))
                for k, v in raw_kwargs.items():
                    if v is None:
                        continue
                    dtype = cls._argument_conversion_map[k]
                    kwargs[k] = _to_tensor(v, dtype=dtype)

        return inputs, kwargs

    def _infer_output_devices(self, *inputs, **args):
        self._init_spec(inputs, args)
        return self._output_devices

    def _pre_call(self, *inputs, **args):
        pass

    def _is_backend_initialized(self):
        return self._op_backend is not None

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
                op = self._legacy_op(name=self._name, device=self._backend, **self._init_args)
                self._op_inst = op
                out = op(*input_nodes, **arg_nodes)
                if isinstance(out, (list, tuple)):
                    spec = out[0].source.spec
                elif isinstance(out, dict):
                    spec = next(iter(out.values())).source.spec
                    self._output_names = tuple(out.keys())
                else:
                    spec = out.source.spec

                self._op_spec = spec

                if isinstance(out, (tuple, list)):
                    self._output_devices = []
                    self._num_outputs = len(out)
                    for o in out:
                        device_type = o.device
                        device_id = None if device_type == "cpu" else self._device.device_id
                        self._output_devices.append(Device(device_type, device_id))
                elif isinstance(out, dict):
                    self._num_outputs = len(out)
                    device_id = self._device.device_id
                    self._output_devices = [
                        Device(o.device, None if o.device == "cpu" else device_id)
                        for o in out.values()
                    ]
                else:
                    self._num_outputs = 1
                    self._output_devices = [
                        Device(out.device, None if out.device == "cpu" else self._device.device_id)
                    ]

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
                        if self._backend == "gpu" or self._backend == "mixed"
                        else dali.types.CPU_ONLY_DEVICE_ID
                    ),
                )
                if self._max_batch_size is None:
                    self._max_batch_size = 1
                self._op_spec.AddArg("max_batch_size", self._max_batch_size)
                self._op_backend = _b._Operator(self._op_spec)

    def _run(self, ctx, *inputs, batch_size=None, **args):
        device_id = ctx.device_id if ctx is not None else None
        device_ctx = Device("gpu", device_id) if device_id is not None else Device("cpu")
        with device_ctx:
            if (
                batch_size is not None
                and self._max_batch_size is not None
                and batch_size > self._max_batch_size
                and self._is_stateful
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
                # clearing the backend in a stateful op would destroy the state
                self._check_compatible(inputs, batch_size, args)

            self._init_backend(ctx, inputs, args)
            workspace = _b._Workspace(ctx._thread_pool, ctx.cuda_stream)
            for i, input in enumerate(inputs):
                workspace.AddInput(self._to_batch(input).evaluate()._storage)
            for name, arg in args.items():
                workspace.AddArgumentInput(name, self._to_batch(arg).evaluate()._storage)
            self._op_backend.SetupAndRun(workspace, batch_size)
            out = workspace.GetOutputs()

            result = out if is_batch else tuple(o[0] for o in out)
            return result if self._output_names is None else dict(zip(self._output_names, result))

    def _to_batch(self, x):
        if not isinstance(x, Batch):
            return Batch([x])
        else:
            return x

    def _set_meta(self, inputs, args):
        self._input_meta = [self._make_meta(input) for input in inputs]
        self._arg_meta = {name: self._make_meta(arg) for name, arg in args.items()}

    def _check_compatible(self, inputs, batch_size, args):
        """Raises an error if the inputs and arguments are not compatible with this op instance."""

        def error_header():
            return (
                f"The invocation of operator {self._display_name} "
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

    # TODO(klecki): Consider making a dataclass
    @staticmethod
    def _make_meta(x):
        if x is None:
            return ()
        is_batch = False
        if isinstance(x, _invocation.Invocation):
            is_batch = x.is_batch
        elif isinstance(x, Batch):
            is_batch = True
        else:
            is_batch = False

        return (
            is_batch,
            x.ndim,
            x.layout,
            x.dtype.type_id,
        )

    @property
    def _display_name(self):
        if "display_name" in self._init_args:
            type_name = self._init_args["display_name"]
        else:
            type_name = self._schema.OperatorName()
        if self._name is not None:
            return f'type_name "{self._name}"'
        else:
            return type_name


# Defaults used by Reader for sharding; must match Reader.__init__ normalization.
_READER_SHARD_DEFAULTS = {"shard_id": 0, "num_shards": 1, "stick_to_shard": False}


class Reader(Operator):
    """Base class for reader operators. Extends Operator with iteration support via next_epoch().

    Readers maintain internal state and can provide samples or batches. Mixing iteration styles
    (samples/batches/direct calls) on the same instance is forbidden."""

    def __init__(
        self,
        batch_size=None,
        name=None,
        device="cpu",
        shard_id=_READER_SHARD_DEFAULTS["shard_id"],
        num_shards=_READER_SHARD_DEFAULTS["num_shards"],
        stick_to_shard=_READER_SHARD_DEFAULTS["stick_to_shard"],
        **kwargs,
    ):
        if name is None:
            name = f"Reader_{id(self)}"
        self._actual_batch_size = batch_size
        self._batch_size = batch_size
        device = _device.device(device)
        # _backend is forwarded via **kwargs, we don't need to touch it here
        self._shard_id = shard_id if shard_id is not None else _READER_SHARD_DEFAULTS["shard_id"]
        self._num_shards = (
            num_shards if num_shards is not None else _READER_SHARD_DEFAULTS["num_shards"]
        )
        self._stick_to_shard = (
            stick_to_shard
            if stick_to_shard is not None
            else _READER_SHARD_DEFAULTS["stick_to_shard"]
        )
        if self._num_shards < 1:
            raise ValueError(
                f"The number of shards must be a positive integer. Got {self._num_shards}."
            )
        if self._shard_id < 0 or self._shard_id >= self._num_shards:
            raise ValueError(
                f"The shard_id={self._shard_id} is invalid. Must be in range "
                + f"[0..{self._num_shards-1}]."
            )
        kwargs["shard_id"] = self._shard_id
        kwargs["num_shards"] = self._num_shards
        kwargs["stick_to_shard"] = self._stick_to_shard
        super().__init__(self._actual_batch_size, name, device, **kwargs)

    @classmethod
    def _get(cls, *_, **__):
        raise RuntimeError("Readers cannot be cached. Construct a new instance instead.")

    def _pre_call(self, *inputs, **args):
        if self._api_type is None:
            self._api_type = "_run"
        elif self._api_type != "_run":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `_run`/`__call__` on the same reader."
            )

    def _run(self, ctx=None, *inputs, **args):
        """
        Runs the reader and obtains one result (batch or sample, depending on `batch_size`).

        Do not call this function directly. Use `__call__` instead.
        """
        if self._api_type is None:
            self._api_type = "_run"
        elif self._api_type != "_run":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `_run`/`__call__` on the same reader."
            )

        return super()._run(ctx, *inputs, **args)

    def next_epoch(self, batch_size=None, ctx: _eval_context.EvalContext | None = None):
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

    def _samples(self, ctx: _eval_context.EvalContext | None = None):
        if self._api_type is None:
            self._api_type = "samples"
        elif self._api_type != "samples":
            raise RuntimeError(
                "Cannot mix `samples`, `batches` and `_run`/`__call__` on the same reader."
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
            padded_size = meta["epoch_size_padded"]
            shards_beg = math.floor(self._shard_id * padded_size / self._num_shards)
            shards_end = math.floor((self._shard_id + 1) * padded_size / self._num_shards)
            while idx < shards_end - shards_beg:
                outputs = super()._run(ctx, batch_size=self._actual_batch_size)
                batch_size = len(
                    outputs[0] if isinstance(outputs, tuple) else next(iter(outputs.values()))
                )
                assert batch_size == self._actual_batch_size
                idx += batch_size
                if isinstance(outputs, tuple):
                    for x in zip(*outputs):
                        outs = tuple(Tensor(o) for o in x)
                        yield outs
                else:
                    names = outputs.keys()
                    for x in zip(*outputs.values()):
                        outs = tuple(Tensor(o) for o in x)
                        yield dict(zip(names, outs))
            if not self._stick_to_shard:
                self._shard_id = (self._shard_id + 1) % self._num_shards

    def _batches(self, batch_size=None, ctx: _eval_context.EvalContext | None = None):
        if self._api_type is None:
            self._api_type = "batches"
        elif self._api_type != "batches":
            raise RuntimeError("Cannot mix samples(), batches() and _run() on the same reader.")

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
            padded_size = meta["epoch_size_padded"]
            shards_beg = math.floor(self._shard_id * padded_size / self._num_shards)
            shards_end = math.floor((self._shard_id + 1) * padded_size / self._num_shards)
            while idx < shards_end - shards_beg:
                outputs = super()._run(ctx, batch_size=batch_size)
                batch_size_returned = batch_size = len(
                    outputs[0] if isinstance(outputs, tuple) else next(iter(outputs.values()))
                )
                assert batch_size_returned == batch_size
                idx += batch_size_returned
                if isinstance(outputs, tuple):
                    yield tuple(Batch(o) for o in outputs)
                else:
                    yield {name: Batch(o) for name, o in outputs.items()}
            if not self._stick_to_shard:
                self._shard_id = (self._shard_id + 1) % self._num_shards


_all_ops = []
_all_functions = []


def _initialize():
    from . import _op_builder

    global _all_ops, _all_functions
    _all_ops, _all_functions = _op_builder.build_operators()
