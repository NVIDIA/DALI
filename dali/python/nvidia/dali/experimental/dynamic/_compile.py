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
import threading
import types
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import nvidia.dali.backend_impl as _b
import nvidia.dali.types as dali_types
from nvidia.dali import fn
from nvidia.dali.external_source import ExternalSource
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
    from ._ops import Operator, Reader


def _nvtx_range(message: str):
    return NVTXRange(message, color=0xB58900, category="compile")


class State(enum.Enum):
    TRACING = enum.auto()
    COMPILED = enum.auto()
    DISABLED = enum.auto()


class CompileSource(NamedTuple):
    """The reader root of the compile graph."""

    num_outputs: int
    output_keys: tuple[str, ...] | None = None
    device: str = "cpu"


class CompileRef(NamedTuple):
    """Reference to one output of a compile graph node."""

    source: "CompileSource | CompileNode"
    output_index: int


@dataclasses.dataclass(eq=False)
class CompileNode:
    """A captured operator call in the compile graph."""

    op_class: type["Operator"]
    backend: str
    inputs: Sequence[CompileRef | Any]
    kwargs: Mapping[str, CompileRef | Any]
    kwarg_casts: dict[str, dali_types.DALIDataType]
    num_outputs: int
    device: Device | None = None
    pipeline_output_offset: int | None = dataclasses.field(default=None, repr=False)


class _CallTrie:
    """Trie keyed by call chain CodeLocs for safe call-site identification."""

    __slots__ = ("children", "node")

    def __init__(self) -> None:
        self.children: dict[CodeLoc, _CallTrie] = {}
        self.node: CompileNode | None = None

    def insert(self, call_chain: CallChain, node: CompileNode) -> None:
        current = self
        for code_loc in call_chain:
            if code_loc not in current.children:
                current.children[code_loc] = _CallTrie()
            current = current.children[code_loc]
        current.node = node

    def find(self, call_chain: CallChain) -> CompileNode | None:
        """Look up a node by call chain tuple (not frame). Returns None if not found."""
        current = self
        for code_loc in call_chain:
            child = current.children.get(code_loc)
            if child is None:
                return None
            current = child
        return current.node

    def lookup(self, start_frame: types.FrameType) -> CompileNode | None:
        """Walk frames to stack exhaustion or stop early if a frame differs."""
        current = self
        frame: types.FrameType | None = start_frame
        while frame is not None:
            child = current.children.get(CodeLoc(frame.f_code, frame.f_lasti))
            if child is None:
                return None
            current = child
            frame = frame.f_back
        return current.node


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
    """Manages compile state for one reader (TRACING → COMPILED or DISABLED)."""

    _tls = threading.local()

    def __init__(self, reader: "Reader", batch_size: int):
        from ._source_analysis import CallSiteAnalyzer

        self.state = State.TRACING
        self.reader = reader
        self.batch_size = batch_size
        self.source: CompileSource | None = None
        self.nodes: list[CompileNode] = []
        self._call_trie = _CallTrie()
        self.pipeline: Pipeline | None = None
        self._pipeline_results: dict[CompileNode, Any] = {}
        self._iteration = 0
        self.analyzer = CallSiteAnalyzer()

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
            raise RuntimeError(
                "Multiple compiled readers active simultaneously is not supported. "
                "Only one reader can use compile=True at a time."
            )
        CompileContext._tls.current = self
        try:
            yield
        finally:
            CompileContext._tls.current = prev

    def init_source(
        self,
        num_outputs: int,
        output_keys: tuple[str, ...] | None = None,
        device: str = "cpu",
    ) -> None:
        if self.source is not None:
            assert self.source.num_outputs == num_outputs
            assert self.source.output_keys == output_keys
            return
        self.source = CompileSource(num_outputs, output_keys, device)

    def make_source_batches(self, tensor_lists: Sequence[Any]) -> tuple[CompiledBatch, ...]:
        assert self.source is not None
        return tuple(
            CompiledBatch(tl, CompileRef(self.source, i), self._iteration)
            for i, tl in enumerate(tensor_lists)
        )

    @staticmethod
    def _compute_kwarg_casts(op: type["Operator"], raw_kwargs: Mapping[str, CompiledBatch | Any]):
        casts: dict[str, dali_types.DALIDataType] = {}
        schema = op._schema
        assert schema is not None

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
        if existing := self._call_trie.find(call_chain):
            if existing.inputs == inputs and existing.kwargs == kwargs:
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
        self._call_trie.insert(call_chain, node)
        return node

    @_nvtx_range("Building pipeline")
    def build_pipeline(self, ctx: "EvalContext") -> None:
        if not self.nodes:
            warnings.warn(
                "compile=True was specified but no operators were captured during tracing. "
                "Falling back to dynamic mode.",
            )
            self.state = State.DISABLED
            return

        self._assign_output_offsets()

        source = self.source
        assert source is not None

        def reader_callback():
            assert self.reader._op_backend is not None
            workspace = _b._Workspace(ctx.thread_pool._create_facade(), ctx.cuda_stream)
            for name, arg in self.reader._process_tensor_args(self.batch_size).items():
                workspace.AddArgumentInput(name, arg.evaluate()._storage)
            self.reader._op_backend.SetupAndRun(workspace, self.batch_size)
            return workspace.GetOutputs()

        try:
            pipe = Pipeline(
                batch_size=self.batch_size,
                num_threads=ctx.num_threads,
                device_id=ctx.device_id,
                prefetch_queue_depth=2,
            )
            with pipe:
                _wire_compile_graph(source, self.nodes, reader_callback)
            pipe.build()
        except Exception:
            self.nodes.clear()
            self._call_trie = _CallTrie()
            self._pipeline_results.clear()
            self.source = None
            self.state = State.DISABLED
            # Reset Reader fields here, the exception propagates past the iterator
            self.reader._compile_mode = None
            self.reader._compiled_iter = None
            raise

        self.pipeline = pipe
        self.state = State.COMPILED

    @_nvtx_range("Running compiled pipeline")
    def run_pipeline(self) -> tuple | dict:
        """Run the compiled pipeline and cache all node results for this iteration."""
        assert self.pipeline is not None and self.source is not None
        self._iteration += 1
        pipeline_outputs = self.pipeline.run()
        source_batches = tuple(
            CompiledBatch(pipeline_outputs[i], CompileRef(self.source, i), self._iteration)
            for i in range(self.source.num_outputs)
        )
        self._pipeline_results.clear()
        for node in self.nodes:
            assert node.pipeline_output_offset is not None
            offset = node.pipeline_output_offset
            batches = tuple(
                CompiledBatch(pipeline_outputs[offset + i], CompileRef(node, i), self._iteration)
                for i in range(node.num_outputs)
            )
            self._pipeline_results[node] = batches[0] if node.num_outputs == 1 else batches
        if self.source.output_keys is not None:
            return dict(zip(self.source.output_keys, source_batches))
        return source_batches

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
        return not isinstance(actual, Batch) and actual == expected

    @_nvtx_range("Getting compiled result")
    def get_compiled_result(
        self,
        frame: types.FrameType,
        inputs: Sequence[Any],
        kwargs: Mapping[str, Any],
        device: Device | None = None,
    ) -> Any | None:
        """Return pre-built result for a known call site, or None."""
        node = self._call_trie.lookup(frame)
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
        return self._pipeline_results.get(node)

    def _assign_output_offsets(self) -> None:
        assert self.source is not None
        offset = self.source.num_outputs
        for node in self.nodes:
            node.pipeline_output_offset = offset
            offset += node.num_outputs


class CompiledEpochIterator:
    """Owns the compile lifecycle for one reader."""

    def __init__(self, reader: "Reader", batch_size: int):
        self._reader = reader
        self._compile_ctx = CompileContext(reader, batch_size)
        self._ctx: "EvalContext | None" = None

    @property
    def compile_context(self) -> CompileContext:
        return self._compile_ctx

    def batches(self, ctx: "EvalContext | None"):
        """Yield batches, tracing on first epoch, compiled thereafter."""
        from . import _eval_context

        reader = self._reader
        reader._require_api_type("batches")

        if ctx is None:
            ctx = _eval_context.EvalContext.current()
        if self._ctx is not None and ctx is not self._ctx:
            raise RuntimeError("Cannot change EvalContext for a compiled reader.")
        self._ctx = ctx
        with ctx:
            if self._compile_ctx.state is State.COMPILED:
                yield from self._batches_compiled()
            else:
                yield from self._batches_tracing(ctx)
        reader._advance_shard()

    def _batches_tracing(self, ctx: "EvalContext"):
        """First epoch: run dynamically, record graph, build pipeline."""
        compile_ctx = self._compile_ctx
        reader = self._reader
        batch_size = compile_ctx.batch_size
        tensor_args = reader._process_tensor_args(batch_size)

        if not reader._op_backend:
            reader._max_batch_size = batch_size
            reader._init_backend(ctx, (), tensor_args)

        epoch_size = reader._shard_epoch_size()
        idx = 0
        first_iteration = True
        while idx < epoch_size:
            outputs = reader._run_unchecked(ctx, batch_size=batch_size, **tensor_args)
            batch_size_returned = reader._output_batch_size(outputs)
            idx += batch_size_returned
            if isinstance(outputs, tuple):
                raw = outputs
                output_keys = None
            else:
                raw = tuple(outputs.values())
                output_keys = tuple(outputs.keys())
            # No reader returns multiple outputs on different devices
            device = "gpu" if isinstance(raw[0], _b.TensorListGPU) else "cpu"
            compile_ctx.init_source(len(raw), output_keys=output_keys, device=device)
            batches = compile_ctx.make_source_batches(raw)
            with compile_ctx.active():
                if isinstance(outputs, tuple):
                    yield batches
                else:
                    yield dict(zip(outputs.keys(), batches))
            if first_iteration:
                first_iteration = False
                compile_ctx.build_pipeline(ctx)
                if compile_ctx.state is State.COMPILED:
                    yield from self._batches_compiled(start_idx=idx)
                    return
                if compile_ctx.state is State.DISABLED:
                    reader._compile_mode = None
                    reader._compiled_iter = None

    def _batches_compiled(self, start_idx: int = 0):
        """Subsequent epochs: yield from compiled pipeline."""
        compile_ctx = self._compile_ctx

        epoch_size = self._reader._shard_epoch_size()
        idx = start_idx
        while idx < epoch_size:
            batches = compile_ctx.run_pipeline()
            idx += self._reader._output_batch_size(batches)
            with compile_ctx.active():
                yield batches


@_nvtx_range("Graph Wiring")
def _wire_compile_graph(
    source: CompileSource,
    nodes: Sequence[CompileNode],
    reader_callback: Callable[[], Iterable],
) -> None:
    """Wire the compile graph into a Pipeline. Must be called inside ``with pipe:``."""
    from ._op_builder import _scalar_decay

    es = ExternalSource(
        source=reader_callback,
        num_outputs=source.num_outputs,
        batch=True,
        device=source.device,
        no_copy=True,
    )
    reader_outs = es()
    assert isinstance(reader_outs, Iterable)

    datanode_map: dict[CompileRef, Any] = {}
    for i, out in enumerate(reader_outs):
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

    outputs = [datanode_map[CompileRef(source, i)] for i in range(source.num_outputs)]
    for node in nodes:
        outputs.extend(datanode_map[CompileRef(node, i)] for i in range(node.num_outputs))
    Pipeline.current().set_outputs(*outputs)


def _compile_intercept(
    fn_call: types.FunctionType, op_class: type["Operator"], op_name: str | None = None
) -> types.FunctionType:
    """Wrap an fn_call to intercept operator calls for transparent pipelining."""
    from ._op_builder import _resolve_backend

    @mark_transparent
    def wrapper(*inputs, batch_size=None, device=None, **raw_kwargs):
        @mark_transparent
        def _call():
            return fn_call(
                *inputs, batch_size=batch_size, device=device, _backend=backend, **raw_kwargs
            )

        device, backend = _resolve_backend(op_class, device, inputs, raw_kwargs, op_name=op_name)
        compile_ctx = CompileContext.current()
        if compile_ctx is None:
            return _call()

        # Resolves past transparent frames (this wrapper, makefun, NVTXRange, fn_call)
        # to the user's call site.
        frame = resolve_callsite_frame()
        if frame is None:
            return _call()

        if compile_ctx.state is State.COMPILED:
            if batch_size is not None and batch_size != compile_ctx.batch_size:
                raise RuntimeError(
                    f"Compiled reader uses batch_size={compile_ctx.batch_size} but operator "
                    f"called with batch_size={batch_size}. Cannot change batch_size in "
                    f"compiled mode."
                )
            if result := compile_ctx.get_compiled_result(frame, inputs, raw_kwargs, device=device):
                return result
            return _call()

        # Run first, classify after, we need the result before we can inspect it
        result = _call()

        if op_class._is_stateful:
            return result

        classification = compile_ctx.analyzer.classify(frame, inputs, raw_kwargs)
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
