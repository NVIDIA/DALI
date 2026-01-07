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

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx


class DaliTabsDirective(Directive):
    """
    Generate sphinx-design tabs.

    Usage:
        .. dali-tabs::

           **Pipeline mode:**

           .. code-block:: python

              # pipeline code

           **Dynamic mode:**

           .. code-block:: python

              # dynamic code
    """

    has_content = True
    required_arguments = 0
    optional_arguments = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pattern: **Tab Name:** followed by content until next **Tab:** or end
        self._pattern = re.compile(
            r"\*\*([^*]+):\*\*\s*\n(.*?)(?=\*\*[^*]+:\*\*|\Z)",
            re.DOTALL,
        )

    def run(self):
        content = "\n".join(self.content)

        matches = self._pattern.findall(content)
        if not matches:
            # Fallback: just parse content normally
            container = nodes.container()
            self.state.nested_parse(
                self.content, self.content_offset, container
            )
            return [container]

        # Build tab-set RST
        tab_set_lines = [".. tab-set::", "   :sync-group: dali-mode", ""]

        for tab_name, tab_content in matches:
            tab_name = tab_name.strip()
            sync_key = tab_name.lower().replace(" ", "-")

            tab_set_lines.append(f"   .. tab-item:: {tab_name}")
            tab_set_lines.append(f"      :sync: {sync_key}")
            tab_set_lines.append("")

            # Indent the content properly for tab-item
            for line in tab_content.strip().split("\n"):
                if line.strip():
                    tab_set_lines.append(f"      {line}")
                else:
                    tab_set_lines.append("")
            tab_set_lines.append("")

        # Parse the generated tab-set RST
        tab_rst = StringList(tab_set_lines)

        container = nodes.container()
        self.state.nested_parse(tab_rst, self.content_offset, container)
        return container.children


def setup(app: Sphinx):
    app.add_directive("dali-tabs", DaliTabsDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
