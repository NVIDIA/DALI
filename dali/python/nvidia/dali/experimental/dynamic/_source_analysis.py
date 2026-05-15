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
import itertools
import linecache
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from ._call_site import CodeLoc
from ._compile import CompileRef
from ._device import Device
from ._type import DType

CodePosition: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SourceCalls:
    by_span: Mapping[CodePosition, ast.Call | None]
    by_line: Mapping[int, tuple[ast.Call, ...]]

    @classmethod
    def from_ast(cls, tree: ast.Module) -> "SourceCalls":
        by_span: dict[CodePosition, ast.Call | None] = {}
        by_line: dict[int, list[ast.Call]] = {}
        collect_spans = sys.version_info >= (3, 11)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            by_line.setdefault(node.lineno, []).append(node)
            if not collect_spans:
                continue

            pos = (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
            if any(x is None for x in pos):
                continue
            pos = cast(CodePosition, pos)
            by_span[pos] = None if pos in by_span else node

        return cls(
            by_span=types.MappingProxyType(by_span),
            by_line=types.MappingProxyType(
                {lineno: tuple(calls) for lineno, calls in by_line.items()}
            ),
        )


_file_cache: dict[str, tuple[object, SourceCalls | None]] = {}


def is_constant_node(node: ast.AST) -> bool:
    """Check if an AST node represents a constant expression."""
    match node:
        case ast.Constant():
            return True
        case ast.UnaryOp(op=ast.USub() | ast.UAdd()):
            return is_constant_node(node.operand)
        case ast.BinOp():
            return is_constant_node(node.left) and is_constant_node(node.right)
        case ast.List() | ast.Tuple():
            return all(is_constant_node(elt) for elt in node.elts)
        case _:
            return False


def is_dali_constant(value: Any) -> bool:
    """Check if a value is a constant defined by DALI (e.g., ndd.int32)"""
    return isinstance(value, (Device, DType, DALIDataType, DALIInterpType, DALIImageType))


def _get_source_calls(filename: str) -> SourceCalls | None:
    """Parse and cache source calls. Returns None for empty/invalid source."""
    lines = linecache.getlines(filename)
    if not lines:
        return None

    # We rely on the linecache entry to make sure that our cache didn't get invalidated
    entry = linecache.cache[filename]
    cached = _file_cache.get(filename)
    # The cache entry is not hashable so we can't use it as a key
    if cached is not None and cached[0] is entry:
        return cached[1]

    try:
        tree = ast.parse("".join(lines), filename=filename)
    except SyntaxError:
        calls = None
    else:
        calls = SourceCalls.from_ast(tree)

    _file_cache[filename] = (entry, calls)
    return calls


def _get_positions(code: types.CodeType, lasti: int) -> CodePosition | None:
    """Return the (start_line, end_line, start_col, end_col) for the bytecode index `lasti`."""
    if sys.version_info < (3, 11) or lasti < 0:
        return None
    pos = next(itertools.islice(code.co_positions(), lasti // 2, None), None)
    if pos is None or any(x is None for x in pos):
        # Some items can be None, e.g. when PYTHONNODEBUGRANGES=1
        return None

    return cast(CodePosition, pos)


def get_call_from_frame(frame: types.FrameType) -> ast.Call | None:
    """Resolve the ``ast.Call`` executing at `frame`'s current instruction, or None"""
    calls = _get_source_calls(frame.f_code.co_filename)
    if calls is None:
        return None
    if pos := _get_positions(frame.f_code, frame.f_lasti):
        return calls.by_span.get(pos)
    candidates = calls.by_line.get(frame.f_lineno, ())
    return candidates[0] if len(candidates) == 1 else None


class CallSiteAnalyzer:
    """Caches AST analysis per CodeLoc and classifies operator arguments."""

    def __init__(self) -> None:
        self._cache: dict[CodeLoc, ast.Call | None] = {}

    def classify(
        self,
        frame: types.FrameType,
        inputs: tuple[Any, ...],
        raw_kwargs: dict[str, Any],
    ) -> tuple[list[CompileRef | Any], dict[str, CompileRef | Any]] | None:
        """Classify all arguments as CompileRef or constant. Returns None if any fails."""
        from ._compile import CompiledBatch

        call = self._get_call_node(frame)
        if call is None:
            return None

        classified_inputs: list[CompileRef | Any] = []
        for i, inp in enumerate(inputs):
            if inp is None:
                classified_inputs.append(None)
                continue
            if isinstance(inp, CompiledBatch):
                classified_inputs.append(inp._compile_ref)
            elif i < len(call.args) and is_constant_node(call.args[i]) or is_dali_constant(inp):
                classified_inputs.append(inp)
            else:
                return None

        kw_nodes = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        classified_kwargs: dict[str, CompileRef | Any] = {}
        for name, value in raw_kwargs.items():
            if value is None:
                continue
            if isinstance(value, CompiledBatch):
                classified_kwargs[name] = value._compile_ref
            elif name in kw_nodes and is_constant_node(kw_nodes[name]) or is_dali_constant(value):
                classified_kwargs[name] = value
            else:
                return None

        return classified_inputs, classified_kwargs

    def _get_call_node(self, frame: types.FrameType) -> ast.Call | None:
        code_loc = CodeLoc(frame.f_code, frame.f_lasti)
        if code_loc not in self._cache:
            self._cache[code_loc] = get_call_from_frame(frame)
        return self._cache[code_loc]
