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

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform


class DaliTabsTransform(SphinxPostTransform):
    """Transform ``.. container:: dali-tabs`` into sphinx-design tab-sets.

    Usage::

        .. container:: dali-tabs

           **Pipeline mode:**

           .. code-block:: python

              # pipeline code

           **Dynamic mode:**

           .. code-block:: python

              # dynamic code
    """

    default_priority = 199

    def run(self) -> None:
        for container in list(self.document.findall(nodes.container)):
            if "dali-tabs" in container.get("classes", []):
                self._transform_to_tabs(container)

    def _transform_to_tabs(self, container: nodes.container) -> None:
        tabs = self._extract_tabs(container)
        if not tabs:
            return

        tab_set = nodes.container(classes=["sd-tab-set", "docutils"])
        tab_set_id = id(container)

        for i, (tab_name, content_nodes) in enumerate(tabs):
            sync_key = tab_name.lower().replace(" ", "-")
            tab_id = f"sd-tab-item-{tab_set_id}-{i}"
            checked = ' checked="checked"' if i == 0 else ""

            tab_set += nodes.raw(
                "",
                f'<input{checked} id="{tab_id}" '
                f'name="sd-tab-set-{tab_set_id}" type="radio">',
                format="html",
            )
            tab_set += nodes.raw(
                "",
                f'<label class="sd-tab-label" data-sync-group="dali-mode" '
                f'data-sync-id="{sync_key}" for="{tab_id}">\n{tab_name}</label>',
                format="html",
            )

            tab_content = nodes.container(
                classes=["sd-tab-content", "docutils"]
            )
            for node in content_nodes:
                tab_content += node.deepcopy()
            tab_set += tab_content

        container.replace_self(tab_set)

    def _extract_tabs(
        self, container: nodes.container
    ) -> list[tuple[str, list[nodes.Node]]]:
        """Extract tabs from container children.

        Looks for ``**Tab Name:**`` patterns (bold text ending with colon)
        followed by content until the next tab header.
        """
        tabs = []
        current_tab = None
        current_content = []

        for child in container.children:
            if self._is_tab_header(child):
                if current_tab is not None:
                    tabs.append((current_tab, current_content))
                current_tab = child.astext()[:-1]  # Remove trailing colon
                current_content = []
            elif current_tab is not None:
                current_content.append(child)

        if current_tab is not None:
            tabs.append((current_tab, current_content))

        return tabs

    def _is_tab_header(self, node: nodes.Node) -> bool:
        """Check if node is a paragraph containing only bold text ending with colon."""
        if not isinstance(node, nodes.paragraph) or len(node.children) != 1:
            return False
        child = node.children[0]
        return isinstance(child, nodes.strong) and child.astext().endswith(":")


def setup(app: Sphinx) -> dict:
    app.add_post_transform(DaliTabsTransform)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
