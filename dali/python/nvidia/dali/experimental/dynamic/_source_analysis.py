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
import inspect
import itertools
import linecache
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import libcst as cst
from libcst import matchers
from libcst.metadata import (
    Assignment,
    CodeRange,
    FunctionScope,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    Scope,
    ScopeProvider,
)
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from ._call_site import CodeLoc
from ._compile import CompiledBatch, CompileRef
from ._device import Device
from ._type import DType

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
class ModuleInfo:
    """Per-file parsed libcst data plus the queries classification needs over it."""

    scope_of_node: Mapping[cst.CSTNode, Scope]  # LibCST ScopeProvider
    parent_of: Mapping[cst.CSTNode, cst.CSTNode]  # LibCST ParentNodeProvider

    calls_by_position: Mapping[tuple[int, int, int, int], cst.Call | None]  # None if ambiguous
    calls_by_line: Mapping[int, tuple[cst.Call, ...]]  # 3.10 fallback for call-site identification

    call_cache: dict[CodeLoc, cst.Call | None] = field(default_factory=dict, repr=False)

    def call_at(self, frame: types.FrameType) -> cst.Call | None:
        """The ``cst.Call`` executing at `frame`'s current instruction, memoized per call site."""
        key = CodeLoc(frame.f_code, frame.f_lasti)
        if key not in self.call_cache:
            self.call_cache[key] = self._resolve_call(frame)
        return self.call_cache[key]

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

    def local_rhs(self, name_node: cst.Name) -> cst.BaseExpression | None:
        """The RHS of `name_node`'s single capturable function-local binding, else None."""
        scope = self.scope_of_node.get(name_node)
        if scope is None:
            return None
        resolved = scope[name_node.value]  # LEGB-resolved; empty set if undefined
        if len(resolved) != 1:
            return None  # rebound / nonlocal-rebind / undefined

        assignment = next(iter(resolved))
        if type(assignment) is not Assignment:
            return None  # excludes ImportAssignment and BuiltinAssignment

        if not isinstance(assignment.scope, FunctionScope):
            return None  # function locals only (excludes global / class / comprehension)
        if isinstance(assignment.node, cst.Param):
            return None  # parameters, not handled for now
        if assignment.scope is not scope:
            return None  # closure, not handled for now
        return self._rhs_for_target(assignment.node)

    def _rhs_for_target(self, target: cst.CSTNode) -> cst.BaseExpression | None:
        match self.parent_of.get(target):
            case cst.AssignTarget(target=cst.Name()) as target:  # `x = v`
                return cast(cst.Assign, self.parent_of.get(target)).value
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


def _get_module_info(filename: str) -> ModuleInfo | None:
    """Parse and cache the ModuleInfo for `filename`.

    Resolving metadata for the whole file is the dominant trace-time cost
    but is paid once per file and amortizes over a multi-epoch run.
    """
    lines = linecache.getlines(filename)
    if not lines:
        return None
    # The linecache entry detects edits; it is not hashable so it can't be the key itself.
    entry = linecache.cache.get(filename)
    # Note: We don't guard for concurrent calls because the function is idempotent anyway.
    if (cached := _file_cache.get(filename)) is not None and cached[0] is entry:
        return cached[1]
    try:
        wrapper = MetadataWrapper(cst.parse_module("".join(lines)), unsafe_skip_copy=True)
        md = wrapper.resolve_many([PositionProvider, ScopeProvider, ParentNodeProvider])

        pos = cast(Mapping[cst.CSTNode, CodeRange], md[PositionProvider])
        by_position: dict[tuple[int, int, int, int], cst.Call | None] = {}
        by_line: dict[int, list[cst.Call]] = {}
        for node in matchers.findall(wrapper.module, matchers.Call()):
            call = cast(cst.Call, node)
            r = pos[call]
            span = (r.start.line, r.end.line, r.start.column, r.end.column)
            by_position[span] = None if span in by_position else call  # seen twice -> ambiguous
            by_line.setdefault(r.start.line, []).append(call)

        info = ModuleInfo(
            scope_of_node=cast(Mapping[cst.CSTNode, Scope], md[ScopeProvider]),
            parent_of=cast(Mapping[cst.CSTNode, cst.CSTNode], md[ParentNodeProvider]),
            calls_by_position=by_position,
            calls_by_line={ln: tuple(calls) for ln, calls in by_line.items()},
        )
    except Exception:
        info = None
    _file_cache[filename] = (entry, info)
    return info


@dataclass(frozen=True, slots=True)
class _Classifier:
    """Per call frame argument classifier.

    Each argument is either an already captured batch, invariant or not capturable.
    """

    module_info: ModuleInfo
    frame: types.FrameType

    def classify(
        self, inputs: tuple[Any, ...], raw_kwargs: dict[str, Any]
    ) -> tuple[list[CompileRef | Any], dict[str, CompileRef | Any]] | None:
        call = self.module_info.call_at(self.frame)
        if call is None or any(a.star for a in call.args):
            return None  # no call node, or caller-side *args/**kwargs
        pos_nodes = [a.value for a in call.args if a.keyword is None]
        kw_nodes = {a.keyword.value: a.value for a in call.args if a.keyword is not None}

        try:
            classified_inputs: list[CompileRef | Any] = []
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
            return None  # an argument is neither a CompiledBatch nor a capturable constant
        return classified_inputs, classified_kwargs

    def _capture_arg(self, node: cst.BaseExpression | None, value: Any) -> CompileRef | Any:
        if isinstance(value, CompiledBatch):
            return value._compile_ref
        if node is not None and self.is_invariant(node):
            return value
        raise _Unresolved

    def is_invariant(self, node: cst.BaseExpression) -> bool:
        match node:
            case cst.BaseNumber() | cst.SimpleString():
                return True
            case cst.Name(value="True" | "False" | "None"):
                return True
            case cst.UnaryOperation(operator=cst.Minus() | cst.Plus(), expression=x):
                return self.is_invariant(x)
            case cst.BinaryOperation(left=left, right=right):
                return self.is_invariant(left) and self.is_invariant(right)
            case cst.NamedExpr(value=value):
                return self.is_invariant(value)  # walrus `(c := v)` evaluates to v
            case cst.List() | cst.Tuple():
                return all(self.is_invariant(e.value) for e in node.elements)
            case cst.Name():
                return self._is_name_invariant(node)
            case cst.Attribute():
                # We can't accept any attributes, even if the base is a local name.
                # Mutability and aliasing makes them hard to reliably track.
                return self._is_dali_chain(node)
        return False

    def _is_name_invariant(self, name_node: cst.Name) -> bool:
        rhs = self.module_info.local_rhs(name_node)
        if rhs is None or not self.is_invariant(rhs):
            return False
        # A named mutable is a live handle the user can alias and mutate.
        # It's hard to prove that they are invariant.
        try:
            return _is_immutable_value(_safe_resolve(name_node, self.frame))
        except _Unresolved:
            return False

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


def classify(
    frame: types.FrameType, inputs: tuple[Any, ...], raw_kwargs: dict[str, Any]
) -> tuple[list[CompileRef | Any], dict[str, CompileRef | Any]] | None:
    """Classify operator args as captured constants / CompileRefs, or None to run eager."""
    mi = _get_module_info(frame.f_code.co_filename)
    if mi is None:
        return None

    classifier = _Classifier(mi, frame)
    return classifier.classify(inputs, raw_kwargs)
