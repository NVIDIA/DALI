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
import linecache
import types
from typing import Any

from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from ._callsite import CodeLoc
from ._compile import CompileRef
from ._device import Device
from ._type import DType


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


def parse_call_from_frame(frame: types.FrameType) -> ast.Call | None:
    """Parse the source line and return the outermost Call node.

    Current limitations:
        - Multiline expressions are not supported
        - When there are multiple calls on the same line, we pick the first one

    The second one requires Python 3.11+ with columns offsets to be done precisely.
    Best effort with fallback to dynamic can be done for Python 3.10.
    """
    source_line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
    if not source_line:
        return None
    try:
        tree = ast.parse(source_line)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return node
    return None


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
            self._cache[code_loc] = parse_call_from_frame(frame)
        return self._cache[code_loc]
