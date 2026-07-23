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
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, NoReturn, Protocol, TypeVar

import numpy as np

import nvidia.dali.types as dali_types
from nvidia.dali import fn
from nvidia.dali.pipeline import Pipeline

from . import random as _random
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
    random_state_ref: "CompileRef | None" = dataclasses.field(default=None, repr=False)


class CompileRef(NamedTuple):
    """Reference to one output of a compile graph node."""

    owner: "CompileSource | CompileNode | CompiledRNG"
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

    def prune(self, nodes: set[CompileNode]) -> None:
        """Remove graph nodes and any branches left empty by the removal."""
        self.nodes = {op: node for op, node in self.nodes.items() if node not in nodes}
        for child in self.children.values():
            child.prune(nodes)
        self.children = {
            code_loc: child
            for code_loc, child in self.children.items()
            if child.nodes or child.children
        }


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


def _wrap_captured_result(
    node: CompileNode,
    result: Batch | tuple[Batch, ...],
    iteration: int,
) -> CompiledBatch | tuple[CompiledBatch, ...]:
    """Wrap an eager result with its compile provenance."""
    is_tuple = isinstance(result, tuple)
    result = result if is_tuple else (result,)
    wrapped = tuple(
        CompiledBatch.from_batch(batch, CompileRef(node, i), iteration)
        for i, batch in enumerate(result)
    )
    return wrapped if is_tuple else wrapped[0]


class CompiledRNG:
    """Manage one RNG across tracing and compiled execution.

    The public RNG advances when the Python body consumes a compiled random call. A backend clone
    feeds the pipeline and may advance ahead due to prefetch. The call index verifies that the
    Python body consumes the prefetched results in trace order.
    """

    __slots__ = (
        "rng",
        "batch_size",
        "nodes",
        "_used_eagerly",
        "_pipeline_rng",
        "_version",
        "_next_call",
    )

    def __init__(self, rng: _random.RNG, batch_size: int):
        self.rng = rng
        self.batch_size = batch_size
        self.nodes: list[CompileNode] = []  # Operators that this rng is assigned to
        self._used_eagerly = False
        self._pipeline_rng = None  # Clone of rng's generator, advanced in parallel
        self._version = 0
        self._next_call = 0

    def record_use(self, node: CompileNode | None) -> None:
        """Record one traced use."""
        if node is None:
            self._used_eagerly = True
            return

        if node.random_state_ref is None:
            node.random_state_ref = CompileRef(self, len(self.nodes))
            self.nodes.append(node)
            return

        # random_state_ref already present: same call site hit twice
        owner = node.random_state_ref.owner
        assert isinstance(owner, CompiledRNG)
        owner._used_eagerly = True
        self._used_eagerly = True

    @property
    def needs_pruning(self) -> bool:
        return self._used_eagerly and bool(self.nodes)

    def prune(self, nodes: set[CompileNode]) -> bool:
        self.nodes = [node for node in self.nodes if node not in nodes]
        for call_index, node in enumerate(self.nodes):
            node.random_state_ref = CompileRef(self, call_index)
        return bool(self.nodes)

    def sync(self) -> None:
        self._pipeline_rng, self._version = self.rng._snapshot_backend()

    def check_version(self) -> None:
        if self._version != self.rng._version:
            raise RuntimeError("The RNG of a compiled operator was modified outside the loop.")

    def consume_call(self, call_index: int, actual_rng: _random.RNG) -> None:
        if self.rng is not actual_rng:
            raise RuntimeError("A compiled operator was called with a different RNG.")
        self.check_version()
        if call_index < self._next_call:
            raise RuntimeError("A compiled random operator may be called only once per step.")
        if call_index != self._next_call:
            raise RuntimeError(
                "Compiled operators sharing an RNG must be called in the same order."
            )

        self.rng.advance(_random._STATE_WORDS)
        self._version = self.rng._version
        self._next_call += 1

    def finish_step(self) -> None:
        if self._next_call != len(self.nodes):
            raise RuntimeError("A compiled random operator was not consumed this step")
        self._next_call = 0

    def _draw_states(self) -> tuple[Any, ...]:
        random_words = (_random._draw_state(self._pipeline_rng.next) for _ in self.nodes)
        return tuple(
            Batch.broadcast(_random._state_tensor(words), self.batch_size).evaluate()._storage
            for words in random_words
        )

    def _wire_source(self) -> tuple:
        return tuple(fn.external_source(self._draw_states, len(self.nodes), device="cpu"))


def _find_nodes_to_prune(
    nodes: Sequence[CompileNode],
    rngs: Sequence[CompiledRNG],
) -> set[CompileNode]:
    """Find nodes made eager by dependencies or shared RNGs."""
    dependents: dict[CompileNode, list[CompileNode]] = {}
    for node in nodes:
        for ref in itertools.chain(node.inputs, node.kwargs.values()):
            if isinstance(ref, CompileRef) and isinstance(ref.owner, CompileNode):
                dependents.setdefault(ref.owner, []).append(node)

    queue = deque(node for rng in rngs if rng.needs_pruning for node in rng.nodes)
    nodes_to_prune: set[CompileNode] = set()

    while queue:
        node = queue.popleft()
        if node in nodes_to_prune:
            continue

        nodes_to_prune.add(node)
        queue.extend(dependents.get(node, ()))

        if node.random_state_ref is not None:
            rng = node.random_state_ref.owner
            assert isinstance(rng, CompiledRNG)
            queue.extend(rng.nodes)

    return nodes_to_prune


class CompileContext:
    """Manages the compile state (TRACING -> COMPILED or DISABLED)."""

    _tls = threading.local()

    def __init__(self, batch_size: int):
        self.state = State.TRACING
        self.batch_size = batch_size
        self.sources: list[CompileSource] = []  # only sources[0] is iterated on
        self.nodes: list[CompileNode] = []
        self.rngs: dict[_random.RNG, CompiledRNG] = {}
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

        if self.state is State.COMPILED:
            with self._invalidate_on_error():
                for rng in self.rngs.values():
                    rng.finish_step()

    def _teardown(self) -> None:
        self.state = State.DISABLED
        for source in self.sources:
            source.compilable._teardown_compile()

    def _fail(self, message: str) -> NoReturn:
        self._teardown()
        raise RuntimeError(message)

    @contextmanager
    def _invalidate_on_error(self):
        try:
            yield
        except RuntimeError:
            self._teardown()
            raise

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
        frame: types.FrameType,
        op_class: type["Operator"],
        inputs: Sequence[Any],
        kwargs: Mapping[str, Any],
        result: Any,
        backend: str,
        device: Device | None,
    ) -> CompileNode | None:
        from ._source_analysis import classify

        classification = classify(frame, inputs, kwargs)
        if classification is None:
            return None

        captured_inputs, captured_kwargs = classification
        call_chain = build_call_chain(frame)
        if existing := self._call_trie.find(call_chain, op_class):
            if (
                existing.inputs == captured_inputs
                and existing.kwargs == captured_kwargs
                and existing.device == device
            ):
                return existing
            return None

        node = CompileNode(
            op_class=op_class,
            backend=backend,
            inputs=captured_inputs,
            kwargs=captured_kwargs,
            kwarg_casts=self._compute_kwarg_casts(op_class, kwargs),
            num_outputs=len(result) if isinstance(result, tuple) else 1,
            device=device,
        )
        self.nodes.append(node)
        self._call_trie.insert(call_chain, op_class, node)
        return node

    def _record_rng_use(self, rng: _random.RNG, captured_node: CompileNode | None) -> None:
        compiled_rng = self.rngs.get(rng)
        if compiled_rng is None:
            compiled_rng = CompiledRNG(rng, self.batch_size)
            self.rngs[rng] = compiled_rng
        compiled_rng.record_use(captured_node)

    def _prune_rng_nodes(self) -> None:
        nodes_to_prune = _find_nodes_to_prune(self.nodes, tuple(self.rngs.values()))
        if nodes_to_prune:
            warnings.warn(
                "An RNG was used both by a capturable random operator and by a non-capturable "
                "random call, or a capturable random call site was used more than once during "
                "tracing. Affected operators and everything depending on them run eagerly."
            )
        self.nodes = [node for node in self.nodes if node not in nodes_to_prune]
        self._call_trie.prune(nodes_to_prune)
        self.rngs = {
            rng: compiled_rng
            for rng, compiled_rng in self.rngs.items()
            if compiled_rng.prune(nodes_to_prune)
        }

    def _assign_output_offsets(self) -> None:
        offset = 0
        for node in itertools.chain(self.sources, self.nodes):
            node.pipeline_output_offset = offset
            offset += node.num_outputs

    @_nvtx_range("Building pipeline")
    def build_pipeline(self, ctx: "EvalContext") -> None:
        if not self.nodes:
            warnings.warn(
                "compile=True was specified but no operators were captured during tracing. "
                "Falling back to dynamic mode.",
            )
        self._prune_rng_nodes()
        if not self.nodes:
            self._teardown()
            return

        self._assign_output_offsets()

        compiled_rngs = tuple(self.rngs.values())
        transferred = False
        try:
            pipe = Pipeline(
                batch_size=self.batch_size,
                num_threads=ctx.num_threads,
                device_id=ctx.device_id,
                prefetch_queue_depth=2,
            )
            with pipe:
                _wire_compile_graph(self.sources, self.nodes, compiled_rngs)
            for rng in compiled_rngs:
                rng.sync()
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

        self._check_rng_versions()
        self._results.clear()
        for owner in itertools.chain(self.sources, self.nodes):
            self._results[owner] = self._wrap_outputs(owner, pipeline_outputs)
        return self.result_for(self.sources[0])

    def _check_rng_versions(self) -> None:
        with self._invalidate_on_error():
            for rng in self.rngs.values():
                rng.check_version()

    def _resync_rngs(self) -> None:
        self._check_rng_versions()
        for rng in self.rngs.values():
            rng.sync()

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
    def _find_compiled_node(
        self,
        frame: types.FrameType,
        op_class: type["Operator"],
        inputs: Sequence[Any],
        kwargs: Mapping[str, Any],
        device: Device | None = None,
    ) -> CompileNode | None:
        """Find a compiled node matching this call."""
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
        return node

    def _resolve_random_call(self, node: CompileNode | None, rng: _random.RNG) -> Any | None:
        """Return a compiled random result, or None to request eager fallback."""
        if node is not None and node.random_state_ref is not None:
            ref = node.random_state_ref
            compiled_rng = ref.owner
            assert isinstance(compiled_rng, CompiledRNG)
            with self._invalidate_on_error():
                compiled_rng.consume_call(ref.output_index, rng)
            return self.result_for(node)

        if rng in self.rngs:
            self._fail("A captured RNG was used by a non-compiled random operator.")
        return None


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
        if ctx.state is State.DISABLED:
            raise RuntimeError("The compiled loop was invalidated and cannot continue.")
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
        if len(self._compile_ctx.sources) > 1 or self._compile_ctx.rngs:
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
        if ctx._iteration > 0:  # a previous epoch's reset discarded prefetched states
            ctx._resync_rngs()
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
def _wire_compile_graph(
    sources: Sequence[CompileSource],
    nodes: Sequence[CompileNode],
    rngs: Sequence[CompiledRNG],
) -> None:
    """Wire the compile graph into a Pipeline. Must be called inside ``with pipe:``."""
    from ._op_builder import _scalar_decay

    datanode_map: dict[CompileRef, Any] = {}
    for source in sources:
        for i, out in enumerate(source.compilable._wire_pipeline(source)):
            datanode_map[CompileRef(source, i)] = out
    for rng in rngs:
        for i, out in enumerate(rng._wire_source()):
            datanode_map[CompileRef(rng, i)] = out

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

        # Inject random state nodes
        if node.random_state_ref is not None:
            kw_nodes["_random_state"] = datanode_map[node.random_state_ref].cpu()

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
    from ._ops import _infer_batch_size

    is_random = op_class._has_random_state_arg

    @mark_transparent
    def wrapper(*inputs, batch_size=None, device=None, **raw_kwargs):
        device, backend = _resolve_backend(op_class, device, inputs, op_name=op_name)
        compile_ctx = CompileContext.current()
        if compile_ctx is None or compile_ctx.state is State.DISABLED:
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

        if is_random:
            rng = _random._resolve_rng(raw_kwargs.get("rng"))
            graph_kwargs = {name: value for name, value in raw_kwargs.items() if name != "rng"}
        else:
            graph_kwargs = raw_kwargs

        if compile_ctx.state is State.COMPILED:
            compile_ctx.check_batch_size(batch_size)
            node = compile_ctx._find_compiled_node(frame, op_class, inputs, graph_kwargs, device)
            if not is_random:
                result = compile_ctx.result_for(node) if node is not None else None
            else:
                if node is not None and batch_size is None:
                    actual_batch_size = _infer_batch_size(*inputs, **graph_kwargs)
                    if actual_batch_size != compile_ctx.batch_size:
                        raise RuntimeError(
                            f"Compiled random operator cannot change batch_size from "
                            f"{compile_ctx.batch_size} to {actual_batch_size}."
                        )
                result = compile_ctx._resolve_random_call(node, rng)
            if result is not None:
                return result
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        # Run first, classify after, we need the result before we can inspect it
        result = fn_call(
            *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
        )
        if not is_random:
            capturable = not op_class._is_stateful
        else:
            outputs = result if isinstance(result, tuple) else (result,)
            capturable = all(
                isinstance(output, Batch) and output.batch_size == compile_ctx.batch_size
                for output in outputs
            )
        node = (
            compile_ctx.record(
                frame, op_class, inputs, graph_kwargs, result, backend=backend, device=device
            )
            if capturable
            else None
        )
        if is_random:
            compile_ctx._record_rng_use(rng, node)

        if node is None:
            return result
        return _wrap_captured_result(node, result, compile_ctx._iteration)

    return wrapper
