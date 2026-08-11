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

import ast
import functools
import inspect
import itertools
import linecache
import sys
import types
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import libcst as cst
from libcst.metadata import (
    Assignment,
    CodeRange,
    FunctionScope,
    MetadataWrapper,
    Scope,
    ScopeProvider,
)
from libcst.metadata.position_provider import PositionProvidingCodegenState

from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from ._call_site import resolve_callsite_frame
from ._capture import CapturedBatch, CaptureRef
from ._device import Device
from ._nvtx import NVTXRange
from ._type import DType
from .capture._invariant import invariant, is_dunder
from .capture._invariant import is_invariant as _is_explicit_invariant

_DALI_CONST_TYPES = (Device, DType, DALIDataType, DALIInterpType, DALIImageType)


def is_dali_constant(value: Any) -> bool:
    return isinstance(value, _DALI_CONST_TYPES)


def _is_dali_anchor(value: Any) -> bool:
    if inspect.ismodule(value):
        return value.__name__ == "nvidia.dali" or value.__name__.startswith("nvidia.dali.")
    return isinstance(value, type) and issubclass(value, _DALI_CONST_TYPES)


class _Unresolved(Exception):
    """Raised to abandon capture of an expression (unresolvable statically, or not capturable)."""


def _is_immutable_value(value: Any) -> bool:
    if isinstance(value, (int, float, complex, bool, str, bytes)) or value is None:
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_immutable_value(item) for item in value)
    return is_dali_constant(value)


def _bind_attribute(base: Any, attr: str) -> Any:
    """base.attr via getattr_static, so no user code (property / __getattr__) runs."""
    try:
        descriptor = inspect.getattr_static(base, attr)
    except AttributeError:
        raise _Unresolved
    match descriptor:
        case staticmethod():
            return descriptor.__func__
        case classmethod():
            owner = base if isinstance(base, type) else type(base)
            return descriptor.__func__.__get__(owner, type(owner))
        case types.FunctionType() if isinstance(base, type) or inspect.ismodule(base):
            return descriptor
        case types.FunctionType():
            return descriptor.__get__(base, type(base))
        case type():
            return descriptor
        case _ if inspect.ismodule(descriptor):
            return descriptor  # submodule (e.g. nvidia.dali.types)
        case _ if is_dali_constant(descriptor):
            return descriptor  # DALI sentinel / enum member
    raise _Unresolved


def _safe_resolve(expr: cst.BaseExpression, frame: types.FrameType) -> Any:
    """Resolve a Name / Attribute / literal without running user code."""
    match expr:
        case cst.BaseNumber(value=v) | cst.SimpleString(value=v):
            return ast.literal_eval(v)
        case cst.Name(value="True"):
            return True
        case cst.Name(value="False"):
            return False
        case cst.Name(value="None"):
            return None
        case cst.Name(value=name):
            for ns in (frame.f_locals, frame.f_globals, frame.f_builtins):
                if name in ns:
                    return ns[name]
            raise _Unresolved
        case cst.Attribute(value=base, attr=cst.Name(value=attr)):
            return _bind_attribute(_safe_resolve(base, frame), attr)
    raise _Unresolved


def _byte_to_char_col(lines: list[str], lineno: int, byte_col: int) -> int | None:
    """Map a UTF-8 byte column to a character column."""
    if not 1 <= lineno <= len(lines):
        return None
    try:
        return len(lines[lineno - 1].encode("utf-8")[:byte_col].decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def _unpack_alignment(
    lhs: Sequence[cst.BaseElement],
    rhs: Sequence[cst.BaseElement],
) -> list[cst.BaseExpression | None] | None:
    """RHS literal bound to each LHS element (None for the *star slot);
    None overall if the RHS itself unpacks"""
    if any(isinstance(e, cst.StarredElement) for e in rhs):
        return None
    rhs_values: list[cst.BaseExpression | None] = [e.value for e in rhs]
    stars = [i for i, e in enumerate(lhs) if isinstance(e, cst.StarredElement)]
    if not stars:
        return rhs_values
    star = stars[0]
    n_tail = len(lhs) - star - 1
    bound: list[cst.BaseExpression | None] = [None] * len(lhs)
    bound[:star] = rhs_values[:star]  # prefix binds left-to-right
    bound[star + 1 :] = rhs_values[len(rhs_values) - n_tail :]  # suffix binds right-to-left
    return bound


def _unpack_bindings(
    lhs: Sequence[cst.BaseElement],
    rhs: Sequence[cst.BaseElement],
) -> dict[cst.CSTNode, cst.BaseExpression]:
    """Record target Name -> bound RHS, recursing into nested tuple/list targets."""

    def _unpack_bindings_impl(lhs: Sequence[cst.BaseElement], rhs: Sequence[cst.BaseElement]):
        bound = _unpack_alignment(lhs, rhs)
        if bound is None:
            return
        for element, expr in zip(lhs, bound):
            if expr is None:  # *star slot: binds a runtime list
                continue
            target = element.value
            if isinstance(target, cst.Name):
                bindings[target] = expr
            elif isinstance(target, (cst.Tuple, cst.List)) and isinstance(
                expr, (cst.Tuple, cst.List)
            ):
                _unpack_bindings_impl(target.elements, expr.elements)

    bindings: dict[cst.CSTNode, cst.BaseExpression] = {}
    _unpack_bindings_impl(lhs, rhs)
    return bindings


@dataclass(frozen=True, slots=True)
class Binding:
    in_scope: bool
    rhs: cst.BaseExpression | None  # None if binding is parameter


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """Per-file parsed libcst data plus the queries classification needs over it."""

    scope_of_node: Mapping[cst.CSTNode, Scope]  # LibCST ScopeProvider
    parent_of: Mapping[cst.CSTNode, cst.CSTNode]  # built by the fused codegen pass

    calls_by_position: Mapping[tuple[int, int, int, int], cst.Call | None]  # None if ambiguous
    calls_by_line: Mapping[int, tuple[cst.Call, ...]]  # 3.10 fallback for call-site identification

    call_cache: dict[tuple[int, int], cst.Call | None] = field(default_factory=dict, repr=False)

    def call_at(self, frame: types.FrameType) -> cst.Call | None:
        """The ``cst.Call`` executing at `frame`'s current instruction, memoized per call site."""
        key = (id(frame.f_code), frame.f_lasti)
        # Use _Unresolved as sentinel value - None is a legitimate cache entry
        if (call := self.call_cache.get(key, _Unresolved)) is not _Unresolved:
            return call
        call = self._resolve_call(frame)
        self.call_cache[key] = call
        return call

    @NVTXRange("_resolve_call", category="source analysis")
    def _resolve_call(self, frame: types.FrameType) -> cst.Call | None:
        code = frame.f_code
        if sys.version_info >= (3, 11):
            # One co_positions tuple per 2-byte code unit.
            pos = next(itertools.islice(code.co_positions(), frame.f_lasti // 2, None), None)
            if pos is not None and all(x is not None for x in pos):
                sl, el, sc, ec = cast(tuple[int, int, int, int], pos)
                lines = linecache.getlines(code.co_filename)
                sc, ec = _byte_to_char_col(lines, sl, sc), _byte_to_char_col(lines, el, ec)
                if sc is None or ec is None:
                    return None
                return self.calls_by_position.get((sl, el, sc, ec))
        candidates = self.calls_by_line.get(frame.f_lineno, ())
        return candidates[0] if len(candidates) == 1 else None

    def binding(self, name_node: cst.Name) -> Binding | None:
        """Return the single function-scope binding of `name_node`, else None."""
        scope = self.scope_of_node.get(name_node)
        if scope is None:
            return None
        resolved = scope[name_node.value]  # LEGB-resolved
        if len(resolved) != 1:
            return None  # rebound / nonlocal

        assignment = next(iter(resolved))
        if type(assignment) is not Assignment:
            return None

        if not isinstance(assignment.scope, FunctionScope):
            return None

        in_scope = assignment.scope is scope
        if isinstance(assignment.node, cst.Param):
            return Binding(in_scope, None)

        rhs = self._rhs_for_target(assignment.node)
        return Binding(in_scope, rhs) if rhs is not None else None

    def _rhs_for_target(self, target: cst.CSTNode) -> cst.BaseExpression | None:
        match self.parent_of.get(target):
            case cst.AssignTarget(target=cst.Name()) as assign_target:  # `x = v`
                return cast(cst.Assign, self.parent_of.get(assign_target)).value
            case cst.AnnAssign(value=value):  # `x: T = v`
                return value
            case cst.NamedExpr(value=value):  # walrus `(x := v)`
                return value
            case cst.Element():  # `x, y = a, b`
                # Climb tuple/list nesting to the AssignTarget. StarredElement is omitted: a target
                # under a *star binds a runtime list, so stopping there rejects it.
                owner = self.parent_of.get(target)
                while isinstance(owner, (cst.Element, cst.Tuple, cst.List)):
                    owner = self.parent_of.get(owner)
                if not isinstance(owner, cst.AssignTarget):  # for/with target, or under a *star
                    return None

                assign = cast(cst.Assign, self.parent_of.get(owner))
                lhs = cast(cst.Tuple | cst.List, owner.target)
                if not isinstance(assign.value, (cst.Tuple, cst.List)):
                    return None

                bindings = _unpack_bindings(lhs.elements, assign.value.elements)
                return bindings.get(target)  # nested-aware, None for a *star slot
        return None


# Keyed by filename, holding the linecache entry for invalidation.
# A file edit rebuilds the ModuleInfo and with it a fresh cache.
_file_cache: dict[str, tuple[object, ModuleInfo | None]] = {}
_code_cache: dict[int, tuple[weakref.ReferenceType[types.CodeType], ModuleInfo]] = {}


class _PositionSink:
    """Minimal object satisfying the ``provider`` protocol the codegen state writes to."""

    __slots__ = ("_computed",)

    def __init__(self) -> None:
        self._computed: dict[cst.CSTNode, CodeRange] = {}


class _FusedCodegenState(PositionProvidingCodegenState):
    """A single codegen pass that yields everything classification reads off the tree.

    ``PositionProvider`` already renders every node to compute syntactic positions, so we
    piggyback the parent map and call collection onto that same traversal. This replaces
    three separate full-tree passes (``PositionProvider``, ``ParentNodeProvider`` and
    ``matchers.findall``) with one; only ``ScopeProvider`` still needs its own pass.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parent_of: dict[cst.CSTNode, cst.CSTNode] = {}
        self.calls: list[cst.Call] = []
        self._node_stack: list[cst.CSTNode] = []

    def before_codegen(self, node: cst.CSTNode) -> None:
        super().before_codegen(node)
        if self._node_stack:
            self.parent_of[node] = self._node_stack[-1]
        self._node_stack.append(node)

    def after_codegen(self, node: cst.CSTNode) -> None:
        super().after_codegen(node)
        self._node_stack.pop()
        if type(node) is cst.Call:
            self.calls.append(node)


def _get_module_info(code: types.CodeType) -> ModuleInfo | None:
    """Parse and cache the ModuleInfo for `filename`.

    Resolving metadata for the whole file is the dominant trace-time cost
    but is paid once per file and amortizes over a multi-epoch run.
    """

    if cached := _code_cache.get(id(code)):
        # if it's alive, we're good - we cannot, however, remove this entry without risking a race
        # condition - it must be kept stale until it's replaced (which happens near the end of
        # this function)
        if cached[0]() is code:
            return cached[1]

    with NVTXRange(f"Get module info for: {code.co_filename}", category="source analysis"):
        filename = code.co_filename

        # getlines() must run *before* we snapshot the linecache entry. For a module the import
        # machinery registered lazily, linecache holds a 1-tuple placeholder that getlines()
        # replaces with the fully-populated entry; snapshotting the entry first would capture the
        # placeholder, so the next call (a different code object in the same file) would see the
        # real entry, mismatch the token, and re-parse the file a second time.
        lines = linecache.getlines(filename)

        # The linecache entry detects edits; it is not hashable so it can't be the key itself.
        entry = linecache.cache.get(filename)
        # Note: We don't guard for concurrent calls because the function is idempotent anyway.
        if (cached := _file_cache.get(filename)) is not None and cached[0] is entry:
            # let the exact-code fast path short-circuit next time
            _code_cache[id(code)] = (weakref.ref(code), cached[1])
            return cached[1]
        try:
            if not lines:
                _code_cache[id(code)] = None
                return None
            with NVTXRange("_get_module_info: Join lines", category="source analysis"):
                src = "".join(lines)
            with NVTXRange("_get_module_info: Parse module", category="source analysis"):
                module = cst.parse_module(src)
            with NVTXRange("_get_module_info: build MetadataWrapper", category="source analysis"):
                wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
            # ScopeProvider (and its ExpressionContextProvider dependency) is the one piece we
            # can't cheaply reproduce; positions, parents and the call list all come from the
            # single fused codegen pass below.
            with NVTXRange("_get_module_info: wrapper.resolve", category="source analysis"):
                md = wrapper.resolve(ScopeProvider)

            with NVTXRange("_get_module_info: Fused codegen", category="source analysis"):
                positions = _PositionSink()
                state = _FusedCodegenState(
                    default_indent=wrapper.module.default_indent,
                    default_newline=wrapper.module.default_newline,
                    provider=positions,
                )
                wrapper.module._codegen(state)

            by_position: dict[tuple[int, int, int, int], cst.Call | None] = {}
            by_line: dict[int, list[cst.Call]] = {}
            with NVTXRange("_get_module_info: Visit calls", category="source analysis"):
                for call in state.calls:
                    r = positions._computed[call]
                    span = (r.start.line, r.end.line, r.start.column, r.end.column)
                    by_position[span] = (
                        None if span in by_position else call
                    )  # seen twice -> ambiguous
                    by_line.setdefault(r.start.line, []).append(call)

            info = ModuleInfo(
                scope_of_node=cast(Mapping[cst.CSTNode, Scope], md),
                parent_of=state.parent_of,
                calls_by_position=by_position,
                calls_by_line={ln: tuple(calls) for ln, calls in by_line.items()},
            )
        except Exception:
            info = None
        file_info = (entry, info)
        code_info = (weakref.ref(code), info)
    _file_cache[filename] = file_info
    _code_cache[id(code)] = code_info
    return info


_call_cache = {}


@dataclass(frozen=True, slots=True)
class CallInfo:
    call: Any
    module_info: ModuleInfo
    meta: dict = field(default_factory=dict)


@NVTXRange("call_info", category="source analysis")
def call_info(frame: types.FrameType) -> CallInfo | None:
    key = (id(frame.f_code), frame.f_lasti)
    if (entry := _call_cache.get(key)) is not None:
        if entry[0]() is frame.f_code:
            return entry[1]

    mi = _get_module_info(frame.f_code)
    if mi is not None:
        call = mi.call_at(frame)
        call_info = CallInfo(call, mi)
    else:
        call_info = None
    _call_cache[key] = (weakref.ref(frame.f_code), call_info)
    return call_info


@dataclass(slots=True)
class _Classifier:
    """Per call frame argument classifier.

    Each argument is either an already captured batch, invariant or not capturable.
    """

    module_info: ModuleInfo | None
    frame: types.FrameType
    required_depth: int = field(default=1, init=False, repr=False)

    def _merge_required_depth(self, child: "_Classifier") -> None:
        current = self.frame
        required_depth = child.required_depth
        while current is not child.frame:
            assert current is not None
            current = current.f_back
            required_depth += 1
        self.required_depth = max(self.required_depth, required_depth)

    def classify(
        self, inputs: tuple[Any, ...], raw_kwargs: dict[str, Any]
    ) -> tuple[list[CaptureRef | Any], dict[str, CaptureRef | Any]] | None:
        call = self.module_info.call_at(self.frame) if self.module_info is not None else None
        source_args = _split_call_args(call) if call is not None else None
        pos_nodes, kw_nodes = source_args or ((), {})

        try:
            classified_inputs: list[CaptureRef | Any] = []
            for i, inp in enumerate(inputs):
                if inp is None:
                    classified_inputs.append(None)
                else:
                    node = pos_nodes[i] if i < len(pos_nodes) else None
                    classified_inputs.append(self._capture_arg(node, inp))
            classified_kwargs = {
                name: self._capture_arg(kw_nodes.get(name), raw)
                for name, raw in raw_kwargs.items()
                if raw is not None
            }
        except _Unresolved:
            return None  # an argument is neither a CapturedBatch nor a capturable constant
        return classified_inputs, classified_kwargs

    @NVTXRange("detect_invariant_args", category="source analysis")
    def detect_invariant_args(
        self, inputs: tuple[Any, ...], raw_kwargs: dict[str, Any]
    ) -> tuple[list[bool], dict[str, bool]] | None:
        call = self.module_info.call_at(self.frame)

        if call is None:
            return None

        split = _split_call_args(call)
        if split is None:
            return None
        pos_nodes, kw_nodes = split

        classified_inputs: list[CaptureRef | Any] = []
        for i in range(min(len(inputs), len(pos_nodes))):
            node = pos_nodes[i]
            classified_inputs.append(self.is_invariant(node, static=True))
        # Go over defaults - they are invariant, because they have to be None
        for i in range(len(pos_nodes), len(inputs)):
            assert inputs[i] is None  #
            classified_inputs.append(True)
        classified_kwargs = {
            name for name in raw_kwargs if self.is_invariant(kw_nodes.get(name), static=True)
        }
        return classified_inputs, classified_kwargs

    def _capture_arg(self, node: cst.BaseExpression | None, value: Any) -> Any:
        if isinstance(value, CapturedBatch):
            return value._capture_ref
        if _is_explicit_invariant(value):
            return value
        if node is not None and self.is_invariant(node, static=False):
            return value
        raise _Unresolved

    def is_invariant(self, node: cst.BaseExpression, static: bool) -> bool:
        match node:
            case cst.BaseNumber() | cst.SimpleString() | cst.Name(value="True" | "False" | "None"):
                return True
            case cst.UnaryOperation(operator=cst.Minus() | cst.Plus(), expression=x):
                return self.is_invariant(x, static=static)
            case cst.BinaryOperation(left=left, right=right):
                return self.is_invariant(left, static=static) and self.is_invariant(
                    right, static=static
                )
            case cst.NamedExpr(value=value):
                return self.is_invariant(value, static=static)  # walrus `(c := v)` evaluates to v
            case cst.List() | cst.Tuple():
                return all(self.is_invariant(e.value, static=static) for e in node.elements)
            case cst.Name():
                return self._is_name_invariant(node, static=static)
            case cst.Attribute():
                # We can't accept any attributes, even if the base is a local name.
                # Mutability and aliasing makes them hard to reliably track.
                is_dali_chain = self._is_dali_chain(node)
                if static:
                    return is_dali_chain
                return is_dali_chain or self._is_explicit_invariant_expr(node)
            case cst.Call() if not static:
                return self._is_explicit_invariant_expr(node)
        return False

    def _is_explicit_invariant_expr(self, expr: cst.BaseExpression) -> bool:
        """Check if an expression corresponds to something wrapped with ndd.invariant"""
        match expr:
            case cst.Name():
                try:
                    return _is_explicit_invariant(_safe_resolve(expr, self.frame))
                except _Unresolved:
                    return False
            case cst.Call():
                try:
                    return _safe_resolve(expr.func, self.frame) is invariant
                except _Unresolved:
                    return False
            case cst.NamedExpr(value=value):
                return self._is_explicit_invariant_expr(value)
            case cst.Attribute(value=base, attr=attr):
                # dunder attributes don't propagate the invariant property
                return not is_dunder(attr.value) and self._is_explicit_invariant_expr(base)
        return False

    def _is_name_invariant(self, name_node: cst.Name, static: bool = False) -> bool:
        try:
            value = _safe_resolve(name_node, self.frame)
        except _Unresolved:
            return False

        if not static and _is_explicit_invariant(value):
            return True

        if self.module_info is None:
            return False

        binding = self.module_info.binding(name_node)
        if binding is None or not self._is_binding_invariant(binding, name_node, static=static):
            return False

        # A named mutable is a live handle the user can alias and mutate.
        # It's hard to prove that they are invariant.
        return _is_immutable_value(value)

    def _is_binding_invariant(
        self, binding: Binding, name_node: cst.Name, static: bool = False
    ) -> bool:
        """True if `name_node`'s binding is invariant (captured name re-roots at live owner)."""
        if binding.in_scope:
            classifier, frame = self, self.frame
        else:
            if static:
                # if it's not assigned in the same scope, then static analysis can't do much
                return False
            if frame := self._live_owner_frame(name_node.value):
                classifier = _Classifier(self.module_info, frame)
            else:
                return True  # owner returned: frozen cell

        if binding.rhs is None:
            if static:
                # Parameters are never statically invariant, since the callee might be called from
                # multiple sites.
                # Theoretically we could track lambdas or local functions, but that's not
                # worth the effort.
                return False
            result = classifier._is_param_invariant(name_node, frame)
        else:
            # punch through local assignments
            result = classifier.is_invariant(binding.rhs, static=static)

        self._merge_required_depth(classifier)
        return result

    def _live_owner_frame(self, name: str) -> types.FrameType | None:
        """Find the live frame owning a closure cell"""
        frame = self.frame.f_back
        while frame is not None:
            if name in frame.f_code.co_cellvars:
                return frame
            frame = frame.f_back
        return None

    def _is_param_invariant(self, name_node: cst.Name, owner_frame: types.FrameType) -> bool:
        """True if parameter `name_node` of `owner_frame` was passed an invariant argument."""
        caller = resolve_callsite_frame(owner_frame.f_back)
        if caller is None:
            return False

        mi = _get_module_info(caller.f_code)  # caller may be in another module
        if mi is None:
            return False

        call = mi.call_at(caller)
        if call is None:
            return False

        child = _Classifier(mi, caller)
        result = child._is_arg_invariant(call, name_node.value, owner_frame.f_code)
        self._merge_required_depth(child)
        return result

    def _is_arg_invariant(
        self, call: cst.Call, param_name: str, callee_code: types.CodeType
    ) -> bool:
        """True if `call` binds `param_name` of `callee_code` to an invariant argument."""
        split = _split_call_args(call)
        if split is None:
            return False
        pos_nodes, kw_nodes = split

        try:
            callable_obj = _safe_resolve(call.func, self.frame)
        except _Unresolved:
            return False
        if not _matches_callee(callable_obj, callee_code):
            return False

        try:
            sig = inspect.signature(callable_obj, follow_wrapped=False)
        except (ValueError, TypeError):
            return False

        param = sig.parameters.get(param_name)
        if param is None or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False

        try:
            bound = sig.bind(*pos_nodes, **kw_nodes)
        except TypeError:
            return False

        if param_name not in bound.arguments:
            return param.default is not inspect.Parameter.empty  # omitted: frozen default
        return self.is_invariant(bound.arguments[param_name], static=False)

    def _is_dali_chain(self, node: cst.Attribute) -> bool:
        """The only supported exceptions for attributes are those
        anchored in nvidia.dali or a DALI enum.
        """
        attrs: list[str] = []
        base: cst.BaseExpression = node
        while isinstance(base, cst.Attribute):
            attrs.append(base.attr.value)
            base = base.value
        if not isinstance(base, cst.Name):
            return False
        try:
            value = _safe_resolve(base, self.frame)  # resolve once, bind attrs incrementally
            anchored = _is_dali_anchor(value)
            if not (anchored or inspect.ismodule(value)):
                return False  # root must be a module or a DALI type, not a user object
            for attr in reversed(attrs):
                value = _bind_attribute(value, attr)
                anchored = anchored or _is_dali_anchor(value)
        except _Unresolved:
            return False
        return anchored and is_dali_constant(value)


def _split_call_args(
    call: cst.Call,
) -> tuple[list[cst.BaseExpression], dict[str, cst.BaseExpression]] | None:
    """Extract positional and keyword arguments from a call expression."""
    if any(a.star for a in call.args):
        return None
    pos = [a.value for a in call.args if a.keyword is None]
    kw = {a.keyword.value: a.value for a in call.args if a.keyword is not None}
    return pos, kw


def _matches_callee(obj: Any, callee_code: types.CodeType) -> bool:
    """Check that `obj` actually matches the function we're expecting to be in"""
    if isinstance(obj, types.MethodType):
        return _matches_callee(obj.__func__, callee_code)
    if isinstance(obj, functools.partial):
        return _matches_callee(obj.func, callee_code)
    return isinstance(obj, types.FunctionType) and obj.__code__ is callee_code


def classify(
    frame: types.FrameType,
    inputs: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
    static: bool = False,
) -> tuple[list[CaptureRef | Any], dict[str, CaptureRef | Any], int] | None:
    """Classify operator args as captured constants / CaptureRefs, or None to run eager."""
    mi = _get_module_info(frame.f_code)
    classifier = _Classifier(mi, frame)
    classification = classifier.classify(inputs, raw_kwargs)
    if classification is None:
        return None
    return (*classification, classifier.required_depth)
