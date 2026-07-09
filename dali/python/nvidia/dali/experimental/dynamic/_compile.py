# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import enum
import itertools
import threading
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, TypeVar

import numpy as np

import nvidia.dali.types as dali_types
from nvidia.dali import fn
from nvidia.dali.pipeline import Pipeline

from ._batch import Batch
from ._call_site import (
    CallChain,
    CodeLoc,
    build_call_chain,
    mark_transparent,
    resolve_callsite_frame,
)
from ._device import Device
from ._nvtx import NVTXRange

if TYPE_CHECKING:
    from ._eval_context import EvalContext
    from ._external_source import ExternalSource  # noqa: F401
    from ._ops import Operator, Reader
    from .random import RNG


def _nvtx_range(message: str):
    return NVTXRange(message, color=0xB58900, category="compile")


class State(enum.Enum):
    TRACING = enum.auto()
    COMPILED = enum.auto()
    DISABLED = enum.auto()


class SupportsCompile(Protocol):
    """Interface for sources that support compiled iteration."""

    _compiled_iter: "CompiledEpochIterator | None"

    def _make_epoch_iterator(self, batch_size: int) -> "CompiledEpochIterator": ...
    def _wire_pipeline(self, source: "CompileSource") -> tuple: ...
    def _transfer_into(self, pipe: Pipeline) -> bool: ...
    def _shape_result(self, source: "CompileSource", batches: tuple) -> Any: ...
    def _teardown_compile(self) -> None: ...


@dataclasses.dataclass(eq=False, slots=True)
class CompileSource:
    """A compile graph source: a transferred reader op or an external_source callback."""

    num_outputs: int
    ctx: "CompileContext"
    compilable: "SupportsCompile"  # the Reader or ExternalSource behind this source
    output_keys: tuple[str, ...] | None = None
    pipeline_output_offset: int | None = None


@dataclasses.dataclass(eq=False)
class CompileNode:
    """A captured operator call in the compile graph."""

    op_class: type["Operator"]
    backend: str
    inputs: Sequence["CompileRef | Any"]
    kwargs: Mapping[str, "CompileRef | Any"]
    kwarg_casts: dict[str, dali_types.DALIDataType]
    num_outputs: int
    device: Device | None = None
    pipeline_output_offset: int | None = dataclasses.field(default=None, repr=False)


class CompileRef(NamedTuple):
    """Reference to one output of a compile graph node."""

    owner: "CompileSource | CompileNode"
    output_index: int


class _CallTrie:
    """Trie keyed by call chain CodeLocs for safe call-site identification."""

    __slots__ = ("children", "nodes")

    def __init__(self) -> None:
        self.children: dict[CodeLoc, _CallTrie] = {}
        self.nodes: dict[type["Operator"], CompileNode] = {}

    def insert(self, call_chain: CallChain, op: type["Operator"], node: CompileNode) -> None:
        current = self
        for code_loc in call_chain:
            current = current.children.setdefault(code_loc, _CallTrie())
        current.nodes[op] = node

    def find(self, call_chain: CallChain, op: type["Operator"]) -> CompileNode | None:
        """Look up a node by call chain tuple (not frame). Returns None if not found."""
        current = self
        for code_loc in call_chain:
            child = current.children.get(code_loc)
            if child is None:
                return None
            current = child
        return current.nodes.get(op)

    def lookup(self, start_frame: types.FrameType, op: type["Operator"]) -> CompileNode | None:
        """Walk frames to stack exhaustion or stop early if a frame differs."""
        current = self
        frame: types.FrameType | None = start_frame
        while frame is not None:
            child = current.children.get(CodeLoc(frame.f_code, frame.f_lasti))
            if child is None:
                return None
            current = child
            frame = frame.f_back
        return current.nodes.get(op)


class CompiledBatch(Batch):
    """A Batch that carries compile-graph provenance."""

    def __init__(self, tl: Any, ref: CompileRef, iteration: int):
        super().__init__(tl)
        self._compile_ref = ref
        self._compile_iteration = iteration

    @classmethod
    def from_batch(cls, batch: Batch, ref: CompileRef, iteration: int) -> "CompiledBatch":
        return cls(batch.evaluate()._storage, ref, iteration)

    def _assign(self, other: Batch) -> None:
        super()._assign(other)
        if isinstance(other, CompiledBatch):
            self._compile_ref = other._compile_ref
            self._compile_iteration = other._compile_iteration
        else:
            # Overwritten with non-compiled data, provenance is invalid
            self._compile_ref = None
            self._compile_iteration = None


class CompileContext:
    """Manages the compile state (TRACING -> COMPILED or DISABLED)."""

    _tls = threading.local()

    def __init__(self, batch_size: int):
        self.state = State.TRACING
        self.batch_size = batch_size
        self.sources: list[CompileSource] = []  # only sources[0] is iterated on
        self.nodes: list[CompileNode] = []
        self._call_trie = _CallTrie()
        self.pipeline: Pipeline | None = None
        self._results: dict[CompileSource | CompileNode, tuple[CompiledBatch, ...]] = {}
        self._iteration = 0
        self._read_this_step: set[CompileSource] = set()  # extra sources pulled this step
        self._root_stopped = False  # sources[0] raised StopIteration: a clean epoch end

    @classmethod
    def current(cls) -> "CompileContext | None":
        return getattr(cls._tls, "current", None)

    @contextmanager
    def active(self):
        if self.state is State.DISABLED:
            yield
            return
        prev = getattr(CompileContext._tls, "current", None)
        if prev is not None and prev is not self:
            raise RuntimeError("Only one compiled loop can be active at a time")
        CompileContext._tls.current = self
        try:
            yield
        finally:
            CompileContext._tls.current = prev

    def check_batch_size(self, batch_size: int | None) -> None:
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError(
                f"Cannot change batch size to {batch_size}, "
                f"the compiled loop uses {self.batch_size}."
            )

    def add_source(
        self,
        num_outputs: int,
        compilable: "SupportsCompile",
        *,
        output_keys: tuple[str, ...] | None = None,
    ) -> CompileSource:
        """Register a graph source (sources[0] is registered first and iterated by the loop)"""
        source = CompileSource(num_outputs, self, compilable, output_keys=output_keys)
        self.sources.append(source)
        return source

    def _wrap_tensor_lists(
        self,
        source: "CompileSource | CompileNode",
        tensor_lists: Sequence,
    ) -> tuple[CompiledBatch, ...]:
        return tuple(
            CompiledBatch(tl, CompileRef(source, i), self._iteration)
            for i, tl in enumerate(tensor_lists)
        )

    def _mark_read(self, source: CompileSource) -> None:
        if source in self._read_this_step:
            raise RuntimeError("An ExternalSource may be read only once per compiled step")
        self._read_this_step.add(source)

    def _mark_stopped(self, source: CompileSource) -> None:
        if source is self.sources[0]:
            self._root_stopped = True

    def _reset_stop(self) -> None:
        # Once per epoch, not per step: prefetch can raise the source's StopIteration
        # one or more run() calls before the step that surfaces it.
        self._root_stopped = False

    def _require_consumed(self) -> None:
        # the executor pulls every source each step, so a skipped one silently drops data
        for source in self.sources[1:]:
            if source not in self._read_this_step:
                self._fail("An ExternalSource was not consumed this step")

    def _teardown(self) -> None:
        self.state = State.DISABLED
        for source in self.sources:
            source.compilable._teardown_compile()

    def _fail(self, message: str):
        self._teardown()
        raise RuntimeError(message)

    @staticmethod
    def _compute_kwarg_casts(op: type["Operator"], raw_kwargs: Mapping[str, CompiledBatch | Any]):
        casts: dict[str, dali_types.DALIDataType] = {}

        for name, data in raw_kwargs.items():
            if not isinstance(data, CompiledBatch):
                continue

            expected_type = op._argument_conversion_map[name].type_id
            if expected_type == data.dtype.type_id:
                continue

            casts[name] = expected_type

        return casts

    @_nvtx_range("Recording operator")
    def record(
        self,
        call_chain: CallChain,
        op_class: type["Operator"],
        backend: str,
        inputs: Sequence[CompileRef | Any],
        kwargs: Mapping[str, CompileRef | Any],
        raw_kwargs: Mapping[str, CompiledBatch | Any],
        num_outputs: int,
        device: Device | None = None,
    ) -> CompileNode | None:
        if existing := self._call_trie.find(call_chain, op_class):
            if (
                existing.inputs == inputs
                and existing.kwargs == kwargs
                and existing.device == device
            ):
                return existing
            return None

        node = CompileNode(
            op_class=op_class,
            backend=backend,
            inputs=inputs,
            kwargs=kwargs,
            kwarg_casts=self._compute_kwarg_casts(op_class, raw_kwargs),
            num_outputs=num_outputs,
            device=device,
        )
        self.nodes.append(node)
        self._call_trie.insert(call_chain, op_class, node)
        return node

    @_nvtx_range("Building pipeline")
    def build_pipeline(self, ctx: "EvalContext") -> None:
        if not self.nodes:
            warnings.warn(
                "compile=True was specified but no operators were captured during tracing. "
                "Falling back to dynamic mode.",
            )
            self._teardown()
            return

        self._assign_output_offsets()

        transferred = False
        try:
            pipe = Pipeline(
                batch_size=self.batch_size,
                num_threads=ctx.num_threads,
                device_id=ctx.device_id,
                prefetch_queue_depth=2,
            )
            with pipe:
                _wire_compile_graph(self.sources, self.nodes)
            for source in self.sources:
                transferred |= source.compilable._transfer_into(pipe)
            pipe.build()
        except Exception as exception:
            self._teardown()
            # Only a transferred reader sets `transferred`; its op now belongs to the failed
            # pipeline and cannot be recovered, so the reader is left disabled.
            if transferred:
                raise RuntimeError(
                    "Failed to build pipeline. Reader is now in invalid state."
                ) from exception
            raise

        self.pipeline = pipe
        self.state = State.COMPILED

    @_nvtx_range("Running compiled pipeline")
    def run_pipeline(self) -> tuple | dict:
        """Run the pipeline, cache results, and return sources[0]'s output.

        ``StopIteration`` propagates for the caller to classify (epoch end or underrun).
        Any other failure invalidates the context.
        """
        assert self.pipeline is not None
        self._iteration += 1
        self._read_this_step.clear()
        try:
            pipeline_outputs = self.pipeline.run()
        except StopIteration:
            raise  # Propagate and let the caller classify
        except Exception:
            self._teardown()
            raise
        self._results.clear()
        for owner in itertools.chain(self.sources, self.nodes):
            self._results[owner] = self._wrap_outputs(owner, pipeline_outputs)
        return self.result_for(self.sources[0])

    def _wrap_outputs(
        self, owner: "CompileSource | CompileNode", pipeline_outputs: Sequence
    ) -> tuple[CompiledBatch, ...]:
        offset = owner.pipeline_output_offset
        assert offset is not None
        outputs = pipeline_outputs[offset : offset + owner.num_outputs]
        return self._wrap_tensor_lists(owner, outputs)

    def result_for(self, owner: "CompileSource | CompileNode") -> Any:
        batches = self._results[owner]
        if isinstance(owner, CompileNode):
            return batches[0] if owner.num_outputs == 1 else batches
        return owner.compilable._shape_result(owner, batches)

    def _matches(self, actual: Any, expected: Any) -> bool:
        """Check if an actual value matches the expected traced value."""
        if isinstance(expected, CompileRef):
            return (
                isinstance(actual, CompiledBatch)
                and actual._compile_ref == expected
                and actual._compile_iteration == self._iteration
            )
        if expected is None:
            return actual is None
        if isinstance(actual, Batch):
            return False

        result = actual == expected
        return result if isinstance(result, bool) else np.all(result).item()

    @_nvtx_range("Getting compiled result")
    def get_compiled_result(
        self,
        frame: types.FrameType,
        op_class: type["Operator"],
        inputs: Sequence[Any],
        kwargs: Mapping[str, Any],
        device: Device | None = None,
    ) -> Any | None:
        """Return pre-built result for a known call site, or None."""
        node = self._call_trie.lookup(frame, op_class)
        if node is None:
            return None
        if device != node.device:
            raise RuntimeError(
                f"Compiled operator was traced with device={node.device} but called with "
                f"device={device}. Cannot change device in compiled mode."
            )
        if len(inputs) != len(node.inputs):
            return None
        if not all(self._matches(a, e) for a, e in zip(inputs, node.inputs)):
            return None
        actual_names = {k for k, v in kwargs.items() if v is not None}
        if actual_names != node.kwargs.keys():
            return None
        if not all(self._matches(kwargs[name], expected) for name, expected in node.kwargs.items()):
            return None
        if node not in self._results:
            return None
        return self.result_for(node)

    def _assign_output_offsets(self) -> None:
        offset = 0
        for node in itertools.chain(self.sources, self.nodes):
            node.pipeline_output_offset = offset
            offset += node.num_outputs


_Compilable = TypeVar("_Compilable", bound=SupportsCompile)


class CompiledEpochIterator(ABC, Generic[_Compilable]):
    """Owns the compile lifecycle for one compilable source."""

    def __init__(self, compilable: _Compilable, batch_size: int):
        self._compilable = compilable
        self._compile_ctx = CompileContext(batch_size)
        self._eval_ctx: "EvalContext | None" = None

    def batches(self, ctx: "EvalContext | None") -> Iterator[CompiledBatch]:
        """Yield one epoch: tracing on the first, compiled thereafter."""
        from ._eval_context import EvalContext

        if ctx is None:
            ctx = EvalContext.current()
        if self._eval_ctx is not None and ctx is not self._eval_ctx:
            raise RuntimeError("Cannot change EvalContext for a compiled loop.")
        self._eval_ctx = ctx

        compiled = self._compile_ctx.state is State.COMPILED
        with ctx:
            yield from (self._compiled() if compiled else self._tracing(ctx))

    def _next_batches(self) -> tuple | dict | None:
        """Run one compiled step. Return the batches, or None at a clean epoch end."""
        try:
            return self._compile_ctx.run_pipeline()
        except StopIteration:
            if self._compile_ctx._root_stopped:
                return None
            self._compile_ctx._fail("A source was exhausted before the iteration ended")

    def _emit_step(self, batches):
        ctx = self._compile_ctx
        try:
            with ctx.active():
                yield batches
        except GeneratorExit:
            self._on_break()
            raise
        ctx._require_consumed()

    @abstractmethod
    def _tracing(self, ctx: "EvalContext") -> Iterator: ...

    @abstractmethod
    def _compiled(self) -> Iterator: ...

    @abstractmethod
    def _on_break(self) -> None: ...


class _ReaderEpochIterator(CompiledEpochIterator["Reader"]):
    def __init__(self, compilable: "Reader", batch_size: int):
        super().__init__(compilable, batch_size)
        self._epoch_size_padded: int | None = None
        self._resume_idx = 0  # batches already emitted during tracing, resumed by _compiled

    def batches(self, ctx: "EvalContext | None"):
        self._compilable._require_api_type("batches")
        yield from super().batches(ctx)
        self._compilable._advance_shard()

    def _epoch_size(self) -> int:
        from ._ops import _shard_size

        reader = self._compilable
        pipeline = self._compile_ctx.pipeline
        assert pipeline is not None

        if self._epoch_size_padded is None:
            meta = pipeline.reader_meta(reader._name)
            self._epoch_size_padded = meta["epoch_size_padded"]

        return _shard_size(self._epoch_size_padded, reader._shard_id, reader._num_shards)

    def _trace_step(self, ctx: "EvalContext", tensor_args: dict) -> tuple[Any, int]:
        """Run one eager reader step, registering the source on first use.
        Return (batches, batch_size).
        """
        reader = self._compilable
        compile_ctx = self._compile_ctx
        outputs = reader._run_unchecked(ctx, batch_size=compile_ctx.batch_size, **tensor_args)

        if isinstance(outputs, tuple):
            output_keys, raw = None, outputs
        else:
            output_keys, raw = zip(*outputs.items())

        if not compile_ctx.sources:
            compile_ctx.add_source(len(raw), reader, output_keys=output_keys)

        batches = compile_ctx._wrap_tensor_lists(compile_ctx.sources[0], raw)
        result = reader._shape_result(compile_ctx.sources[0], batches)
        return result, reader._output_batch_size(outputs)

    def _tracing(self, ctx: "EvalContext"):
        compile_ctx = self._compile_ctx
        reader = self._compilable
        batch_size = compile_ctx.batch_size
        tensor_args = reader._process_tensor_args(batch_size)

        if not reader._op_backend:
            reader._max_batch_size = batch_size
            reader._init_backend(ctx, (), tensor_args)

        epoch_size = reader._shard_epoch_size()
        if epoch_size == 0:
            return

        value, idx = self._trace_step(ctx, tensor_args)  # step 0 records the graph
        yield from self._emit_step(value)

        compile_ctx.build_pipeline(ctx)
        if compile_ctx.state is State.COMPILED:
            self._resume_idx = idx
            yield from self._compiled()
            return

        while idx < epoch_size:  # build disabled: finish the epoch eagerly
            value, count = self._trace_step(ctx, tensor_args)
            idx += count
            with compile_ctx.active():
                yield value

    def _compiled(self):
        epoch_size = self._epoch_size()
        idx = self._resume_idx
        self._resume_idx = 0

        while idx < epoch_size:
            batches = self._next_batches()
            assert batches is not None
            idx += self._compilable._output_batch_size(batches)
            yield from self._emit_step(batches)

    def _on_break(self):
        # consumer aborted mid-step, extra sources already advanced, fail safe
        if len(self._compile_ctx.sources) > 1:
            self._compile_ctx._teardown()


class _ExternalSourceEpochIterator(CompiledEpochIterator["ExternalSource"]):
    def _tracing(self, ctx: "EvalContext"):
        es = self._compilable
        try:
            first = es._trace_pull(self._compile_ctx, self._compile_ctx.batch_size)
        except StopIteration:
            es._teardown_compile()  # empty source: leave the instance unbound and reusable
            return

        yield from self._emit_step(first)

        self._compile_ctx.build_pipeline(ctx)
        if self._compile_ctx.state is State.COMPILED:
            yield from self._compiled()
            return

        assert self._compile_ctx.state is State.DISABLED
        # first batch already yielded above; finish the epoch eagerly
        while True:
            try:
                yield es._eager_call(batch_size=self._compile_ctx.batch_size)
            except StopIteration:
                return

    def _compiled(self):
        ctx = self._compile_ctx
        ctx._reset_stop()
        while (batches := self._next_batches()) is not None:
            yield from self._emit_step(batches)
        assert ctx.pipeline is not None
        ctx.pipeline.reset()

    def _on_break(self) -> None:
        self._compile_ctx._teardown()


def make_iterator(compilable: SupportsCompile, batch_size: int) -> CompiledEpochIterator:
    """Return ``compilable._compiled_iter``, creating it or rejecting a batch_size change"""
    if compilable._compiled_iter is None:
        compilable._compiled_iter = compilable._make_epoch_iterator(batch_size)
    elif compilable._compiled_iter._compile_ctx.batch_size != batch_size:
        raise ValueError(
            f"Cannot change batch_size from "
            f"{compilable._compiled_iter._compile_ctx.batch_size} to {batch_size}"
        )
    return compilable._compiled_iter


@_nvtx_range("Graph Wiring")
def _wire_compile_graph(sources: Sequence[CompileSource], nodes: Sequence[CompileNode]) -> None:
    """Wire the compile graph into a Pipeline. Must be called inside ``with pipe:``."""
    from ._op_builder import _scalar_decay

    datanode_map: dict[CompileRef, Any] = {}
    for source in sources:
        for i, out in enumerate(source.compilable._wire_pipeline(source)):
            datanode_map[CompileRef(source, i)] = out

    for node in nodes:
        positional = [
            datanode_map[x] if isinstance(x, CompileRef) else _scalar_decay(x)
            for x in node.inputs
            if x is not None
        ]
        kw_nodes = {k: datanode_map[v] for k, v in node.kwargs.items() if isinstance(v, CompileRef)}
        kw_scalars = {
            k: _scalar_decay(v) for k, v in node.kwargs.items() if not isinstance(v, CompileRef)
        }

        # Cast kwargs when necessary
        for name, dtype in node.kwarg_casts.items():
            kw_nodes[name] = fn.cast(kw_nodes[name], dtype=dtype)
        # All kwargs need to be on the CPU
        for name, kw_node in kw_nodes.items():
            kw_nodes[name] = kw_node.cpu()

        op = node.op_class._legacy_op(device=node.backend, **kw_scalars)
        out = op(*positional, **kw_nodes)

        if node.num_outputs == 1:
            datanode_map[CompileRef(node, 0)] = out
        else:
            for i, o in enumerate(out):
                datanode_map[CompileRef(node, i)] = o

    outputs = []
    for node in itertools.chain(sources, nodes):
        outputs.extend(datanode_map[CompileRef(node, i)] for i in range(node.num_outputs))
    Pipeline.current().set_outputs(*outputs)


def _compile_intercept(
    fn_call: types.FunctionType, op_class: type["Operator"], op_name: str | None = None
) -> types.FunctionType:
    """Wrap an fn_call to intercept operator calls for transparent pipelining."""
    from ._op_builder import _resolve_backend
    from ._source_analysis import classify

    @mark_transparent
    def wrapper(*inputs, batch_size=None, device=None, **raw_kwargs):
        device, backend = _resolve_backend(op_class, device, inputs, op_name=op_name)
        compile_ctx = CompileContext.current()
        if compile_ctx is None:
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        # Resolves past transparent frames (this wrapper, makefun, NVTXRange, fn_call)
        # to the user's call site.
        frame = resolve_callsite_frame(depth_hint=2)
        if frame is None:
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        if compile_ctx.state is State.COMPILED:
            compile_ctx.check_batch_size(batch_size)
            result = compile_ctx.get_compiled_result(
                frame, op_class, inputs, raw_kwargs, device=device
            )
            if result is not None:
                return result
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        # Run first, classify after, we need the result before we can inspect it
        result = fn_call(
            *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
        )

        if op_class._is_stateful:
            return result

        classification = classify(frame, inputs, raw_kwargs)
        if classification is None:
            return result

        classified_inputs, classified_kwargs = classification
        results = result if isinstance(result, tuple) else (result,)
        node = compile_ctx.record(
            call_chain=build_call_chain(frame),
            op_class=op_class,
            backend=backend,
            inputs=classified_inputs,
            kwargs=classified_kwargs,
            raw_kwargs=raw_kwargs,
            num_outputs=len(results),
            device=device,
        )
        if node is None:
            return result

        new_results = tuple(
            CompiledBatch.from_batch(batch, CompileRef(node, i), compile_ctx._iteration)
            for i, batch in enumerate(results)
        )
        return new_results[0] if not isinstance(result, tuple) else new_results

    return wrapper
