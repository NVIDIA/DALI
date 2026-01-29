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

import re

from sphinx.application import Sphinx

# Pattern to match container:: dali-tabs blocks
CONTAINER_PATTERN = re.compile(
    r"^\.\. container:: dali-tabs\n((?:[ ]{3,}[^\n]*\n|\n)*)",
    re.MULTILINE,
)

# Pattern to match **Tab Name:** headers
TAB_HEADER_PATTERN = re.compile(r"^[ ]*\*\*([^*]+):\*\*\s*$", re.MULTILINE)


def _transform_container_to_tabset(match: re.Match) -> str:
    """
    Transform a ``.. container:: dali-tabs`` block into ``.. tab-set::`` RST.
    See README.rst for an example of usage.
    """
    content = match.group(1)

    # Split by tab headers: [before, name1, content1, name2, content2, ...]
    parts = TAB_HEADER_PATTERN.split(content)
    if len(parts) < 3:
        return match.group(0)  # No tabs found, return unchanged

    lines = [".. tab-set::", "   :sync-group: dali-mode", ""]

    for i in range(1, len(parts), 2):
        tab_name = parts[i].strip()
        tab_content = parts[i + 1] if i + 1 < len(parts) else ""
        sync_key = tab_name.lower().replace(" ", "-")

        lines.append(f"   .. tab-item:: {tab_name}")
        lines.append(f"      :sync: {sync_key}")
        lines.append("")

        # Re-indent: content has 3-space indent from container, add 3 more for tab-item
        for line in tab_content.rstrip().split("\n"):
            if line.strip():
                lines.append(f"   {line}")
            else:
                lines.append("")
        lines.append("")

    return "\n".join(lines)


def include_read_handler(
    app: Sphinx,
    relative_path: str,
    parent_docname: str,
    content: list[str],
) -> None:
    """Transform container:: dali-tabs in included files."""
    if not content:
        return

    text = content[0]
    if "dali-tabs" not in text:
        return
    transformed = CONTAINER_PATTERN.sub(_transform_container_to_tabset, text)
    if transformed != text:
        content[0] = transformed


def setup(app: Sphinx):
    app.connect("include-read", include_read_handler)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
