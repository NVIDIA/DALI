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

import nvidia.dali.experimental.dynamic as ndd
import re


_graph_regex = re.compile(r".*(^|[^A-Za-z0-9_])[Gg]raph([ .,)]|$).*")


def _check_no_pipeline_mode_wording(s, schema):
    assert "TensorList" not in s, f"TensorList found in docs for {schema.Name()}:\n{s}"
    assert not _graph_regex.match(s), f'"Graph" found in the docs for {schema.Name()}:\n{s}'


def should_skip(x):
    return x._schema.IsDocHidden() or x._schema.IsDocPartiallyHidden() or x._schema.IsInternal()


def test_function_docs_present():
    assert ndd._ops._all_functions  # not empty
    for f in ndd._ops._all_functions:
        if should_skip(f):
            continue
        assert len(f.__doc__) > 20, f._schema.Name()


def test_function_docs_no_tensor_list():
    assert ndd._ops._all_functions  # not empty
    for f in ndd._ops._all_functions:
        if should_skip(f):
            continue
        _check_no_pipeline_mode_wording(f.__doc__, f._schema)


def test_op_docs_present():
    assert ndd._ops._all_ops  # not empty
    for c in ndd._ops._all_ops:
        if should_skip(c):
            continue
        assert len(c.__init__.__doc__) > 20, c._schema.Name()
        assert len(c.__call__.__doc__) > 20, c._schema.Name()


def test_op_docs_no_tensor_list():
    assert ndd._ops._all_ops  # not empty
    for c in ndd._ops._all_ops:
        if should_skip(c):
            continue
        _check_no_pipeline_mode_wording(c.__init__.__doc__, c._schema)
        _check_no_pipeline_mode_wording(c.__call__.__doc__, c._schema)
