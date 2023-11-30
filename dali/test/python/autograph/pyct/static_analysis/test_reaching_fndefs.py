# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for reaching_fndefs module."""

import unittest

from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import cfg
from nvidia.dali._autograph.pyct import naming
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names
from nvidia.dali._autograph.pyct import transformer
from nvidia.dali._autograph.pyct.static_analysis import activity
from nvidia.dali._autograph.pyct.static_analysis import reaching_definitions
from nvidia.dali._autograph.pyct.static_analysis import reaching_fndefs


class ReachingFndefsAnalyzerTest(unittest.TestCase):
    def _parse_and_analyze(self, test_fn):
        # TODO(mdan): Use a custom FunctionTransformer here.
        node, source = parser.parse_entity(test_fn, future_features=())
        entity_info = transformer.EntityInfo(
            name=test_fn.__name__,
            source_code=source,
            source_file=None,
            future_features=(),
            namespace={},
        )
        node = qual_names.resolve(node)
        namer = naming.Namer({})
        ctx = transformer.Context(entity_info, namer, None)
        node = activity.resolve(node, ctx)
        graphs = cfg.build(node)
        node = reaching_definitions.resolve(node, ctx, graphs)
        node = reaching_fndefs.resolve(node, ctx, graphs)
        return node

    def assertHasFnDefs(self, node):
        anno.getanno(node, anno.Static.DEFINED_FNS_IN)
