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

"""Prepare the custom-operator example inside the sanitizer test checkout."""

import json
from pathlib import Path
import sys


def replace_checked(source, old, new, count):
    if source.count(old) == count:
        return source.replace(old, new)
    if source.count(new) == count:
        return source
    raise RuntimeError(f"Expected {count} occurrences of {old!r} in custom-operator example")


def prepare(examples_dir):
    example = examples_dir / "custom_operations/custom_operator"
    notebook_path = example / "create_a_custom_operator.ipynb"
    cmake_path = example / "customdummy/CMakeLists.txt"
    notebook = json.loads(notebook_path.read_text())
    build_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "CLICOLOR_FORCE=0" in "".join(cell["source"])
    ]
    if len(build_cells) != 1:
        raise RuntimeError("Expected exactly one custom-operator build cell")

    # NVCC's TCMalloc conflicts with ASan. Only DALI's Python queries need the preload.
    source = "".join(build_cells[0]["source"])
    source = replace_checked(
        source,
        "CLICOLOR_FORCE=0 cmake ..",
        'CLICOLOR_FORCE=0 LD_EXTRA_PRELOAD="$LD_PRELOAD" LD_PRELOAD= cmake ..',
        1,
    )
    source = replace_checked(source, "  make", "  LD_PRELOAD= make", 1)
    build_cells[0]["source"] = source.splitlines(keepends=True)
    cmake = replace_checked(
        cmake_path.read_text(),
        "COMMAND python -c",
        'COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=$ENV{LD_EXTRA_PRELOAD}" python -c',
        2,
    )

    # Validate both transformations before changing either input file.
    cmake_path.write_text(cmake)
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    prepare(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
