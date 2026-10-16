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
from .capture._invariant import is_invariant, unwrap_invariant, unwrap_invariants

if TYPE_CHECKING:
    from ._eval_context import EvalContext
    from ._external_source import ExternalSource  # noqa: F401
    from ._ops import Operator, Reader


def _nvtx_range(message: str):
    return NVTXRange(message, color=0xB58900, category="capture")


class State(enum.Enum):
    TRACING = enum.auto()
    CAPTURED = enum.auto()
    DISABLED = enum.auto()


class SupportsCapture(Protocol):
    """Interface for sources that support capture-mode iteration."""

    _captured_iter: "CapturedEpochIterator | None"

    def _make_epoch_iterator(self, batch_size: int) -> "CapturedEpochIterator": ...
    def _wire_pipeline(self, source: "CaptureSource") -> tuple: ...
    def _transfer_into(self, pipe: Pipeline) -> bool: ...
    def _shape_result(self, source: "CaptureSource", batches: tuple) -> Any: ...
    def _teardown_capture(self) -> None: ...


@dataclasses.dataclass(eq=False, slots=True)
class CaptureSource:
    """A capture graph source: a transferred reader op or an external_source callback."""

    num_outputs: int
    ctx: "CaptureContext"
    capturable: "SupportsCapture"  # the Reader or ExternalSource behind this source
    output_keys: tuple[str, ...] | None = None
    pipeline_output_offset: int | None = None


@dataclasses.dataclass(eq=False)
class CaptureNode:
    """A captured operator call in the capture graph."""

    op_class: type["Operator"]
    backend: str
    inputs: Sequence["CaptureRef | Any"]
    kwargs: Mapping[str, "CaptureRef | Any"]
    kwarg_casts: dict[str, dali_types.DALIDataType]
    num_outputs: int
    device: Device | None = None
    pipeline_output_offset: int | None = dataclasses.field(default=None, repr=False)
    random_state_ref: "CaptureRef | None" = dataclasses.field(default=None, repr=False)


def _value_matches(actual: Any, expected: Any) -> bool:
    if is_invariant(expected):
        if not is_invariant(actual):
            raise RuntimeError(
                "An argument marked with ndd.capture.invariant when captured must remain marked."
            )
        return True

    if expected is None:
        return actual is None

    try:
        result = actual == expected
        return result if isinstance(result, bool) else np.all(result).item()
    except Exception:
        return False


class CaptureRef(NamedTuple):
    """Reference to one output of a capture graph node."""

    owner: "CaptureSource | CaptureNode | CapturedRNG"
    output_index: int


def _common_prefix_len(a: CallChain, b: CallChain) -> int:
    return sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(a, b)))


class _CallTrie:
    """Trie keyed by call chain CodeLocs for safe call-site identification."""

    __slots__ = ("children", "nodes")

    def __init__(self) -> None:
        self.children: dict[CodeLoc, _CallTrie] = {}
        self.nodes: dict[type["Operator"], CaptureNode] = {}

    def insert(self, call_chain: CallChain, op: type["Operator"], node: CaptureNode) -> None:
        current = self
        for code_loc in call_chain:
            current = current.children.setdefault(code_loc, _CallTrie())
        current.nodes[op] = node

    def find(self, call_chain: CallChain, op: type["Operator"]) -> CaptureNode | None:
        """Look up a node by call chain tuple (not frame). Returns None if not found."""
        current = self
        for code_loc in call_chain:
            child = current.children.get(code_loc)
            if child is None:
                return None
            current = child
        return current.nodes.get(op)

    def compact(self, required_depths: Mapping[CaptureNode, int]) -> "_CallTrie":
        """Return a trie using the shortest proof-preserving, operator-unique prefixes."""
        entries: dict[type["Operator"], list[tuple[CallChain, CaptureNode]]] = {}
        pending: list[tuple[_CallTrie, CallChain]] = [(self, ())]
        while pending:
            current, call_chain = pending.pop()
            for op, node in current.nodes.items():
                entries.setdefault(op, []).append((call_chain, node))
            pending.extend(
                (child, call_chain + (code_loc,)) for code_loc, child in current.children.items()
            )

        compacted = _CallTrie()
        for op, op_entries in entries.items():
            for call_chain, node in op_entries:
                shared = max(
                    (
                        _common_prefix_len(call_chain, other_chain)
                        for other_chain, other_node in op_entries
                        if other_node is not node
                    ),
                    default=0,
                )
                depth = max(required_depths[node], shared + 1)
                assert depth <= len(call_chain)
                compacted.insert(call_chain[:depth], op, node)

        return compacted

    def lookup(self, start_frame: types.FrameType, op: type["Operator"]) -> CaptureNode | None:
        """Walk frames until the operator's compacted terminal is reached."""
        current = self
        frame: types.FrameType | None = start_frame
        while frame is not None:
            child = current.children.get((frame.f_code, frame.f_lasti))
            if child is None:
                return None
            current = child
            node = current.nodes.get(op)
            if node is not None:
                return node
            frame = frame.f_back
        return None


class CapturedBatch(Batch):
    """A Batch that carries capture-graph provenance."""

    def __init__(self, tl: Any, ref: CaptureRef, iteration: int):
        super().__init__(tl)
        self._capture_ref = ref
        self._capture_iteration = iteration

    @classmethod
    def from_batch(cls, batch: Batch, ref: CaptureRef, iteration: int) -> "CapturedBatch":
        return cls(batch.evaluate()._storage, ref, iteration)

    def _assign(self, other: Batch) -> None:
        super()._assign(other)
        if isinstance(other, CapturedBatch):
            self._capture_ref = other._capture_ref
            self._capture_iteration = other._capture_iteration
        else:
            # Overwritten with uncaptured data, provenance is invalid
            self._capture_ref = None
            self._capture_iteration = None


def _wrap_captured_result(
    node: CaptureNode,
    result: Batch | tuple[Batch, ...],
    iteration: int,
) -> CapturedBatch | tuple[CapturedBatch, ...]:
    """Wrap an eager result with its capture provenance."""
    is_tuple = isinstance(result, tuple)
    result = result if is_tuple else (result,)
    wrapped = tuple(
        CapturedBatch.from_batch(batch, CaptureRef(node, i), iteration)
        for i, batch in enumerate(result)
    )
    return wrapped if is_tuple else wrapped[0]


def _raise_rng_desync() -> NoReturn:
    raise RuntimeError(
        "The RNG of a captured operator was used unexpectedly: the number of random draws per "
        "iteration changed or the RNG was modified outside the loop"
    )


class CapturedRNG:
    """Schedule one RNG's states across tracing and pipeline execution.

    A captured operator must see the words the eager path would have reached, but the pipeline
    draws iterations ahead of the body. The schedule is therefore predicted from the tracing
    iteration: `period` words per iteration, each captured call at a fixed offset within it.
    """

    __slots__ = (
        "rng",
        "batch_size",
        "calls",
        "period",
        "_version",
        "_clone",
        "_in_flight",
        "_start_pos",
    )

    def __init__(self, rng: _random.RNG, batch_size: int, start_pos: int):
        self.rng = rng
        self.batch_size = batch_size

        self.calls: list[tuple[CaptureNode, int]] = []  # node, and where in an iteration it draws
        self.period = 0
        self._version = rng._version
        self._clone = None  # the pipeline draws its states from this copy of the generator
        self._in_flight = 0  # iterations scheduled but not yet consumed
        self._start_pos = start_pos  # where the current iteration begins

    def record_call(self, node: CaptureNode) -> None:
        node.random_state_ref = CaptureRef(self, len(self.calls))
        self.calls.append((node, self.rng._draws - _random._STATE_WORDS - self._start_pos))

    def sync(self) -> None:
        """Start scheduling, now that the traced iteration has given us the period."""
        self.period = self.rng._draws - self._start_pos
        self._clone, self._version = self.rng._snapshot_backend()
        self._start_pos = self.rng._draws
        self._in_flight = 0

    def resync(self) -> None:
        """Resync at epoch boundary"""
        self._clone, _ = self.rng._snapshot_backend()
        self._in_flight = 0

    def begin_step(self) -> None:
        """Take up the iteration whose outputs are about to be consumed."""
        if self.rng._draws != self._start_pos or self.rng._version != self._version:
            _raise_rng_desync()

        if not self._in_flight:
            _raise_rng_desync()
        self._in_flight -= 1

    def consume_call(self, call_index: int, actual_rng: _random.RNG) -> bool:
        """True to use the pipeline's state for this call, False to draw eagerly instead."""
        expected = self._start_pos + self.calls[call_index][1]
        if self.rng._draws > expected:
            return False  # already drawn: the site ran twice, or its first run went eager
        if actual_rng is not self.rng or actual_rng._version != self._version:
            _raise_rng_desync()
        if actual_rng._draws != expected:
            _raise_rng_desync()
        self.rng.advance(_random._STATE_WORDS)
        return True

    def finish_step(self) -> None:
        if self.rng._draws != self._start_pos + self.period or self.rng._version != self._version:
            _raise_rng_desync()
        self._start_pos += self.period

    def _draw_states(self) -> tuple[Any, ...]:
        """Draw the states for one iteration.

        The clone enters on the first word that iteration draws and leaves on the first word of
        the next one, so only the gaps between offsets are ever needed.
        """
        states = []
        drawn = 0
        for _, offset in self.calls:
            if skip := offset - drawn:  # offsets increase, so never negative
                self._clone.skipahead(skip)
            words = _random._draw_state(self._clone.next)
            drawn = offset + _random._STATE_WORDS
            states.append(
                Batch.broadcast(_random._state_tensor(words), self.batch_size).evaluate()._storage
            )
        if tail := self.period - drawn:
            self._clone.skipahead(tail)  # onto the next iteration's first word
        self._in_flight += 1
        return tuple(states)

    def _wire_source(self) -> tuple:
        return tuple(
            fn.external_source(
                self._draw_states,
                num_outputs=len(self.calls),
                device="cpu",
            )
        )


class CaptureContext:
    """Manages the capture state (TRACING -> CAPTURED or DISABLED)."""

    _tls = threading.local()

    def __init__(self, batch_size: int):
        self.state = State.TRACING
        self.batch_size = batch_size
        self.sources: list[CaptureSource] = []  # only sources[0] is iterated on
        self.nodes: list[CaptureNode] = []
        self.rngs: dict[_random.RNG, CapturedRNG] = {}
        self._call_trie = _CallTrie()
        self._required_depths: dict[CaptureNode, int] = {}
        self.pipeline: Pipeline | None = None
        self._results: dict[CaptureSource | CaptureNode, tuple[CapturedBatch, ...]] = {}
        self._iteration = 0
        self._read_this_step: set[CaptureSource] = set()  # extra sources pulled this step
        self._root_stopped = False  # sources[0] raised StopIteration: a clean epoch end

    @classmethod
    def current(cls) -> "CaptureContext | None":
        return getattr(cls._tls, "current", None)

    @contextmanager
    def active(self):
        if self.state is State.DISABLED:
            yield
            return
        prev = getattr(CaptureContext._tls, "current", None)
        if prev is not None and prev is not self:
            raise RuntimeError("Only one capture-mode loop can be active at a time")
        CaptureContext._tls.current = self
        try:
            yield
        finally:
            CaptureContext._tls.current = prev

    def check_batch_size(self, batch_size: int | None) -> None:
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError(
                f"Cannot change batch size to {batch_size}, "
                f"the capture-mode loop uses {self.batch_size}."
            )

    def add_source(
        self,
        num_outputs: int,
        capturable: "SupportsCapture",
        *,
        output_keys: tuple[str, ...] | None = None,
    ) -> CaptureSource:
        """Register a graph source (sources[0] is registered first and iterated by the loop)"""
        source = CaptureSource(num_outputs, self, capturable, output_keys=output_keys)
        self.sources.append(source)
        return source

    def __del__(self):
        if self.pipeline:
            self.pipeline._shutdown()

    def _wrap_tensor_lists(
        self,
        source: "CaptureSource | CaptureNode",
        tensor_lists: Sequence,
    ) -> tuple[CapturedBatch, ...]:
        return tuple(
            CapturedBatch(tl, CaptureRef(source, i), self._iteration)
            for i, tl in enumerate(tensor_lists)
        )

    def _mark_read(self, source: CaptureSource) -> None:
        if source in self._read_this_step:
            raise RuntimeError("An ExternalSource may be read only once per capture-mode step")
        self._read_this_step.add(source)

    def _mark_stopped(self, source: CaptureSource) -> None:
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

        if self.state is State.CAPTURED:
            with self._invalidate_on_error():
                for rng in self.rngs.values():
                    rng.finish_step()

    def _teardown(self) -> None:
        self.state = State.DISABLED
        for source in self.sources:
            source.capturable._teardown_capture()

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
    def _compute_kwarg_casts(op: type["Operator"], raw_kwargs: Mapping[str, CapturedBatch | Any]):
        casts: dict[str, dali_types.DALIDataType] = {}

        for name, data in raw_kwargs.items():
            if not isinstance(data, CapturedBatch):
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
    ) -> CaptureNode | None:
        from ._source_analysis import classify

        classification = classify(frame, inputs, kwargs)
        if classification is None:
            return None

        captured_inputs, captured_kwargs, required_depth = classification
        call_chain = build_call_chain(frame)
        if existing := self._call_trie.find(call_chain, op_class):
            if (
                len(existing.inputs) != len(captured_inputs)
                or existing.kwargs.keys() != captured_kwargs.keys()
                or existing.device != device
            ):
                return None
            if not all(_value_matches(a, e) for a, e in zip(captured_inputs, existing.inputs)):
                return None
            if not all(
                _value_matches(captured_kwargs[name], e) for name, e in existing.kwargs.items()
            ):
                return None
            node = existing
        else:
            node = CaptureNode(
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

        self._required_depths[node] = max(self._required_depths.get(node, 0), required_depth)
        return node

    def _rng_on_draw(self, rng: _random.RNG) -> None:
        """Anchor an RNG at its first traced draw. Fires for bare generator calls too."""
        if rng not in self.rngs:
            self.rngs[rng] = CapturedRNG(rng, self.batch_size, rng._draws)

    def _record_rng_use(
        self,
        rng: _random.RNG,
        captured_node: CaptureNode,
        frame: types.FrameType,
    ) -> None:
        if captured_node.random_state_ref is not None:
            # Warning points to the user's call site
            warnings.warn_explicit(
                "A random operator call site runs more than once per iteration. Only the first "
                "call each iteration uses the captured pipeline; the rest run eagerly.",
                UserWarning,
                frame.f_code.co_filename,
                frame.f_lineno,
            )
            return
        self.rngs[rng].record_call(captured_node)

    def _drop_unused_rngs(self) -> None:
        """Discard the generators the body only drew from directly; they schedule nothing."""
        self.rngs = {rng: captured for rng, captured in self.rngs.items() if captured.calls}

    def _assign_output_offsets(self) -> None:
        offset = 0
        for node in itertools.chain(self.sources, self.nodes):
            node.pipeline_output_offset = offset
            offset += node.num_outputs

    @_nvtx_range("Building pipeline")
    def build_pipeline(self, ctx: "EvalContext") -> None:
        if not self.nodes:
            warnings.warn(
                "capture=True was specified but no operators were captured during tracing. "
                "Falling back to dynamic mode.",
            )
        self._drop_unused_rngs()
        if not self.nodes:
            self._teardown()
            return

        self._call_trie = self._call_trie.compact(self._required_depths)
        self._required_depths.clear()
        self._assign_output_offsets()

        captured_rngs = tuple(self.rngs.values())
        transferred = False
        try:
            pipe = Pipeline(
                batch_size=self.batch_size,
                num_threads=ctx.num_threads,
                device_id=ctx.device_id,
                prefetch_queue_depth=2,
            )
            with pipe:
                _wire_capture_graph(self.sources, self.nodes, captured_rngs)
            for rng in captured_rngs:
                rng.sync()
            for source in self.sources:
                transferred |= source.capturable._transfer_into(pipe)
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
        self.state = State.CAPTURED

    @_nvtx_range("Running captured pipeline")
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

        with self._invalidate_on_error():
            for rng in self.rngs.values():
                rng.begin_step()

        self._results.clear()
        for owner in itertools.chain(self.sources, self.nodes):
            self._results[owner] = self._wrap_outputs(owner, pipeline_outputs)
        return self.result_for(self.sources[0])

    def _resync_rngs(self) -> None:
        with self._invalidate_on_error():
            for rng in self.rngs.values():
                rng.resync()

    def _wrap_outputs(
        self, owner: "CaptureSource | CaptureNode", pipeline_outputs: Sequence
    ) -> tuple[CapturedBatch, ...]:
        offset = owner.pipeline_output_offset
        assert offset is not None
        outputs = pipeline_outputs[offset : offset + owner.num_outputs]
        return self._wrap_tensor_lists(owner, outputs)

    def result_for(self, owner: "CaptureSource | CaptureNode") -> Any:
        batches = self._results[owner]
        if isinstance(owner, CaptureNode):
            return batches[0] if owner.num_outputs == 1 else batches
        return owner.capturable._shape_result(owner, batches)

    def _matches(self, actual: Any, expected: Any) -> bool:
        """Check if an actual value matches the expected traced value."""
        if type(expected) is CaptureRef:
            actual = unwrap_invariant(actual)
            return (
                isinstance(actual, CapturedBatch)
                and actual._capture_ref == expected
                and actual._capture_iteration == self._iteration
            )
        if isinstance(actual, Batch):
            return False
        return _value_matches(actual, expected)

    @_nvtx_range("Getting captured result")
    def _find_captured_node(
        self,
        frame: types.FrameType,
        op_class: type["Operator"],
        inputs: Sequence[Any],
        kwargs: Mapping[str, Any],
        device: Device | None = None,
    ) -> CaptureNode | None:
        """Find a captured node matching this call."""
        node = self._call_trie.lookup(frame, op_class)
        if node is None:
            return None
        if device != node.device:
            raise RuntimeError(
                f"Captured operator was traced with device={node.device} but called with "
                f"device={device}. Cannot change device in capture mode."
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

    def _resolve_random_call(self, node: CaptureNode | None, rng: _random.RNG) -> Any | None:
        """Return a captured random result, or None to request eager fallback."""
        if node is not None and node.random_state_ref is not None:
            ref = node.random_state_ref
            captured_rng = ref.owner
            assert isinstance(captured_rng, CapturedRNG)
            with self._invalidate_on_error():
                if captured_rng.consume_call(ref.output_index, rng):
                    return self.result_for(node)

        # The eager call draws where the schedule already expects it to.
        return None


def _note_rng_draw(self: _random.RNG) -> None:
    """`RNG._on_draw`, installed below."""
    ctx = CaptureContext.current()
    if ctx is not None and ctx.state is State.TRACING:
        ctx._rng_on_draw(self)


# Intercept on `RNG` rather than in `_capture_intercept`, which bare generator RNG calls bypass.
_random.RNG._on_draw = _note_rng_draw


_Capturable = TypeVar("_Capturable", bound=SupportsCapture)


class CapturedEpochIterator(ABC, Generic[_Capturable]):
    """Owns the capture lifecycle for one capturable source."""

    def __init__(self, capturable: _Capturable, batch_size: int):
        self._capturable = capturable
        self._capture_ctx = CaptureContext(batch_size)
        self._eval_ctx: "EvalContext | None" = None

    def batches(self, ctx: "EvalContext | None") -> Iterator[CapturedBatch]:
        """Yield one epoch: tracing on the first, pipeline execution thereafter."""
        from ._eval_context import EvalContext

        if ctx is None:
            ctx = EvalContext.current()
        if self._eval_ctx is not None and ctx is not self._eval_ctx:
            raise RuntimeError("Cannot change EvalContext for a capture-mode loop.")
        self._eval_ctx = ctx

        captured = self._capture_ctx.state is State.CAPTURED
        with ctx:
            yield from (self._captured() if captured else self._tracing(ctx))

    def _next_batches(self) -> tuple | dict | None:
        """Run one pipeline step. Return the batches, or None at a clean epoch end."""
        try:
            return self._capture_ctx.run_pipeline()
        except StopIteration:
            if self._capture_ctx._root_stopped:
                return None
            self._capture_ctx._fail("A source was exhausted before the iteration ended")

    def _emit_step(self, batches):
        ctx = self._capture_ctx
        try:
            with ctx.active():
                yield batches
        except GeneratorExit:
            self._on_break()
            raise
        if ctx.state is State.DISABLED:
            raise RuntimeError("The capture-mode loop was invalidated and cannot continue.")
        ctx._require_consumed()

    @abstractmethod
    def _tracing(self, ctx: "EvalContext") -> Iterator: ...

    @abstractmethod
    def _captured(self) -> Iterator: ...

    @abstractmethod
    def _on_break(self) -> None: ...


class _ReaderEpochIterator(CapturedEpochIterator["Reader"]):
    def __init__(self, capturable: "Reader", batch_size: int):
        super().__init__(capturable, batch_size)
        self._epoch_size_padded: int | None = None
        self._resume_idx = 0  # batches already emitted during tracing, resumed by _captured

    def batches(self, ctx: "EvalContext | None"):
        self._capturable._require_api_type("batches")
        yield from super().batches(ctx)
        self._capturable._advance_shard()

    def _epoch_size(self) -> int:
        from ._ops import _shard_size

        reader = self._capturable
        pipeline = self._capture_ctx.pipeline
        assert pipeline is not None

        if self._epoch_size_padded is None:
            meta = pipeline.reader_meta(reader._name)
            self._epoch_size_padded = meta["epoch_size_padded"]

        return _shard_size(self._epoch_size_padded, reader._shard_id, reader._num_shards)

    def _trace_step(self, ctx: "EvalContext", tensor_args: dict) -> tuple[Any, int]:
        """Run one eager reader step, registering the source on first use.
        Return (batches, batch_size).
        """
        reader = self._capturable
        capture_ctx = self._capture_ctx
        outputs = reader._run_unchecked(ctx, batch_size=capture_ctx.batch_size, **tensor_args)

        if isinstance(outputs, tuple):
            output_keys, raw = None, outputs
        else:
            output_keys, raw = zip(*outputs.items())

        if not capture_ctx.sources:
            capture_ctx.add_source(len(raw), reader, output_keys=output_keys)

        batches = capture_ctx._wrap_tensor_lists(capture_ctx.sources[0], raw)
        result = reader._shape_result(capture_ctx.sources[0], batches)
        return result, reader._output_batch_size(outputs)

    def _tracing(self, ctx: "EvalContext"):
        capture_ctx = self._capture_ctx
        reader = self._capturable
        batch_size = capture_ctx.batch_size
        tensor_args = reader._process_tensor_args(batch_size)

        if not reader._op_backend:
            reader._max_batch_size = batch_size
            reader._init_backend(ctx, (), tensor_args)

        epoch_size = reader._shard_epoch_size()
        if epoch_size == 0:
            return

        value, idx = self._trace_step(ctx, tensor_args)  # step 0 records the graph
        yield from self._emit_step(value)

        capture_ctx.build_pipeline(ctx)
        if capture_ctx.state is State.CAPTURED:
            self._resume_idx = idx
            yield from self._captured()
            return

        while idx < epoch_size:  # build disabled: finish the epoch eagerly
            value, count = self._trace_step(ctx, tensor_args)
            idx += count
            with capture_ctx.active():
                yield value

    def _captured(self):
        epoch_size = self._epoch_size()
        idx = self._resume_idx
        self._resume_idx = 0

        while idx < epoch_size:
            batches = self._next_batches()
            assert batches is not None
            idx += self._capturable._output_batch_size(batches)
            yield from self._emit_step(batches)

    def _on_break(self):
        # consumer aborted mid-step, extra sources already advanced, fail safe
        if len(self._capture_ctx.sources) > 1 or self._capture_ctx.rngs:
            self._capture_ctx._teardown()


class _ExternalSourceEpochIterator(CapturedEpochIterator["ExternalSource"]):
    def _tracing(self, ctx: "EvalContext"):
        es = self._capturable
        try:
            first = es._trace_pull(self._capture_ctx, self._capture_ctx.batch_size)
        except StopIteration:
            es._teardown_capture()  # empty source: leave the instance unbound and reusable
            return

        yield from self._emit_step(first)

        self._capture_ctx.build_pipeline(ctx)
        if self._capture_ctx.state is State.CAPTURED:
            yield from self._captured()
            return

        assert self._capture_ctx.state is State.DISABLED
        # first batch already yielded above; finish the epoch eagerly
        while True:
            try:
                yield es._eager_call(batch_size=self._capture_ctx.batch_size)
            except StopIteration:
                return

    def _captured(self):
        ctx = self._capture_ctx
        ctx._reset_stop()
        if ctx._iteration > 0:  # a previous epoch's reset discarded prefetched states
            ctx._resync_rngs()
        while (batches := self._next_batches()) is not None:
            yield from self._emit_step(batches)
        assert ctx.pipeline is not None
        ctx.pipeline.reset()

    def _on_break(self) -> None:
        self._capture_ctx._teardown()


def make_iterator(capturable: SupportsCapture, batch_size: int) -> CapturedEpochIterator:
    """Return ``capturable._captured_iter``, creating it or rejecting a batch_size change"""
    if capturable._captured_iter is None:
        capturable._captured_iter = capturable._make_epoch_iterator(batch_size)
    elif capturable._captured_iter._capture_ctx.batch_size != batch_size:
        raise ValueError(
            f"Cannot change batch_size from "
            f"{capturable._captured_iter._capture_ctx.batch_size} to {batch_size}"
        )
    return capturable._captured_iter


@_nvtx_range("Graph Wiring")
def _wire_capture_graph(
    sources: Sequence[CaptureSource],
    nodes: Sequence[CaptureNode],
    rngs: Sequence[CapturedRNG],
) -> None:
    """Wire the capture graph into a Pipeline. Must be called inside ``with pipe:``."""
    from ._op_builder import _scalar_decay

    datanode_map: dict[CaptureRef, Any] = {}
    for source in sources:
        for i, out in enumerate(source.capturable._wire_pipeline(source)):
            datanode_map[CaptureRef(source, i)] = out
    for rng in rngs:
        for i, out in enumerate(rng._wire_source()):
            datanode_map[CaptureRef(rng, i)] = out

    for node in nodes:
        positional = [
            datanode_map[x] if isinstance(x, CaptureRef) else _scalar_decay(x)
            for x in map(unwrap_invariants, node.inputs)
            if x is not None
        ]
        kw_nodes, kw_scalars = {}, {}
        for name, value in node.kwargs.items():
            value = unwrap_invariants(value)
            if isinstance(value, CaptureRef):
                kw_nodes[name] = datanode_map[value]
            elif value is not None:
                kw_scalars[name] = _scalar_decay(value)

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
            datanode_map[CaptureRef(node, 0)] = out
        else:
            for i, o in enumerate(out):
                datanode_map[CaptureRef(node, i)] = o

    outputs = []
    for node in itertools.chain(sources, nodes):
        outputs.extend(datanode_map[CaptureRef(node, i)] for i in range(node.num_outputs))
    Pipeline.current().set_outputs(*outputs)


def _capture_intercept(
    fn_call: types.FunctionType, op_class: type["Operator"], op_name: str | None = None
) -> types.FunctionType:
    """Wrap an fn_call to intercept operator calls for transparent pipelining."""
    from ._op_builder import _resolve_backend
    from ._ops import _infer_batch_size

    is_random = op_class._has_random_state_arg

    @mark_transparent
    def wrapper(*inputs, batch_size=None, device=None, **raw_kwargs):
        batch_size = unwrap_invariant(batch_size)
        device, backend = _resolve_backend(op_class, device, inputs, op_name=op_name)
        capture_ctx = CaptureContext.current()
        if capture_ctx is None or capture_ctx.state is State.DISABLED:
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        # Resolves past transparent frames (this wrapper, makefun, NVTXRange, fn_call)
        # to the user's call site.
        frame = resolve_callsite_frame(depth_hint=2)
        if frame is None:
            return fn_call(
                *inputs,
                batch_size=batch_size,
                device=device,
                _backend=backend,
                _caller_frame=frame,
                **raw_kwargs,
            )

        if is_random:
            rng = _random._resolve_rng(raw_kwargs.get("rng"))
            graph_kwargs = {name: value for name, value in raw_kwargs.items() if name != "rng"}
        else:
            graph_kwargs = raw_kwargs

        if capture_ctx.state is State.CAPTURED:
            capture_ctx.check_batch_size(batch_size)
            node = capture_ctx._find_captured_node(frame, op_class, inputs, graph_kwargs, device)
            if not is_random:
                result = capture_ctx.result_for(node) if node is not None else None
            else:
                if node is not None and batch_size is None:
                    actual_batch_size = _infer_batch_size(*inputs, **graph_kwargs)
                    if actual_batch_size != capture_ctx.batch_size:
                        raise RuntimeError(
                            f"Captured random operator cannot change batch_size from "
                            f"{capture_ctx.batch_size} to {actual_batch_size}."
                        )
                result = capture_ctx._resolve_random_call(node, rng)
            if result is not None:
                return result
            return fn_call(
                *inputs,
                batch_size=batch_size,
                device=device,
                _backend=backend,
                _caller_frame=frame,
                **raw_kwargs,
            )

        # Run first, classify after, we need the result before we can inspect it
        result = fn_call(
            *inputs,
            batch_size=batch_size,
            device=device,
            _backend=backend,
            _caller_frame=frame,
            **raw_kwargs,
        )
        if not is_random:
            can_capture = not op_class._is_stateful
        else:
            outputs = result if isinstance(result, tuple) else (result,)
            can_capture = all(
                isinstance(output, Batch) and output.batch_size == capture_ctx.batch_size
                for output in outputs
            )
        node = (
            capture_ctx.record(
                frame, op_class, inputs, graph_kwargs, result, backend=backend, device=device
            )
            if can_capture
            else None
        )
        if is_random and node is not None:
            capture_ctx._record_rng_use(rng, node, frame)

        if node is None:
            return result
        return _wrap_captured_result(node, result, capture_ctx._iteration)

    return wrapper
