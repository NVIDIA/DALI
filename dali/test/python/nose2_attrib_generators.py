# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Custom nose2 plugin to filter generator test functions by attributes
before they are called (preventing imports of optional dependencies or other code execution).

This plugin monkey-patches the Generators plugin's _testsFromGeneratorFunc
method to check attributes before calling generator functions.
"""
from nose2.events import Plugin
import logging

log = logging.getLogger(__name__)


class AttributeGeneratorFilter(Plugin):
    """Filter generator functions by attributes before calling them."""

    configSection = "attrib-generators"
    alwaysOn = True

    def __init__(self):
        super().__init__()
        self._patched = False

    def _get_attrib_plugin(self):
        """Get the attrib plugin from the session."""
        for plugin in self.session.plugins:
            if plugin.__class__.__name__ == "AttributeSelector":
                return plugin
        return None

    def _build_attribs_list(self, attrib_plugin):
        """Build the attribs list from the attrib plugin's -A configuration.

        This replicates the logic from AttributeSelector.moduleLoadedSuite
        for -A filters only (not -E eval filters).
        """
        attribs = []

        # Handle -A (attribute) filters
        for attr in attrib_plugin.attribs:
            attr_group = []
            for attrib in attr.strip().split(","):
                if not attrib:
                    continue
                items = attrib.split("=", 1)
                if len(items) > 1:
                    # "name=value"
                    key, value = items
                else:
                    key = items[0]
                    if key[0] == "!":
                        # "!name"
                        key = key[1:]
                        value = False
                    else:
                        # "name"
                        value = True
                attr_group.append((key, value))
            attribs.append(attr_group)

        return attribs

    def _matches_attrib_filter(self, test_func, attrib_plugin):
        """Check if test_func matches the attribute filter from attrib plugin."""
        if not attrib_plugin:
            return True

        if not attrib_plugin.attribs:
            return True

        # Build attribs list using attrib plugin's logic
        attribs = self._build_attribs_list(attrib_plugin)

        if not attribs:
            return True

        # Use the plugin's validateAttrib method
        return attrib_plugin.validateAttrib(test_func, attribs)

    def _patch_generator_plugin(self):
        """Monkey-patch the Generators plugin to check attributes first."""
        if self._patched:
            return

        # Find the Generators plugin
        gen_plugin = None
        for plugin in self.session.plugins:
            if plugin.__class__.__name__ == "Generators":
                gen_plugin = plugin
                break

        if not gen_plugin:
            log.warning("Could not find Generators plugin to patch")
            return

        # Save original method
        original_tests_from_gen = gen_plugin._testsFromGeneratorFunc
        attrib_filter_self = self

        # Create patched method
        def patched_tests_from_gen(event, obj):
            """Check attributes before calling generator function."""
            attrib_plugin = attrib_filter_self._get_attrib_plugin()

            # Check if generator function matches attribute filter
            if not attrib_filter_self._matches_attrib_filter(obj, attrib_plugin):
                log.debug(f"Skipping generator {obj.__name__} due to attribute filter")
                return []  # Return empty list

            # Call original method
            return original_tests_from_gen(event, obj)

        # Monkey-patch it
        gen_plugin._testsFromGeneratorFunc = patched_tests_from_gen
        self._patched = True
        log.debug("Patched Generators plugin to check attributes")

    def handleArgs(self, event):
        """Patch right after argument handling, before test discovery."""
        self._patch_generator_plugin()
