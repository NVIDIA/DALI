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

import os
from contextlib import contextmanager
from operator import attrgetter

from ndd_utils import _is_captured
from nose2.tools import params
from nose_utils import assert_warns
from test_utils import get_dali_extra_path, load_test_operator_plugin

import nvidia.dali.backend as _b
from nvidia.dali.experimental import dynamic as ndd
from nvidia.dali.experimental.dynamic import _op_builder

load_test_operator_plugin()
# ndd builds its operators once, on import, and plugin loading doesn't rebuild them - ndd may
# already have been imported by another test module, so register the test operators explicitly.
for _schema_name in (
    "DynDeprecationWarningOp",
    "sub__DynDeprecationWarningOp",
    "sub__DynDeprecationReplacementOp",
):
    _op_builder.build_fn_wrapper(_op_builder.build_operator_class(_b.GetSchema(_schema_name)))

images_root = os.path.join(get_dali_extra_path(), "db", "single", "jpeg")


@params(
    ("dyn_deprecation_warning_op", "", "Additional message"),
    ("sub.dyn_deprecation_warning_op", "sub.dyn_deprecation_replacement_op", "Another message"),
)
def test_deprecation_warning_names_dynamic_api(op_path, replacement, message):
    op = attrgetter(op_path)(ndd)

    glob = f"WARNING: `nvidia.dali.experimental.dynamic.{op_path}` is now deprecated."
    if replacement:
        glob += f" Use `nvidia.dali.experimental.dynamic.{replacement}` instead."
    glob += f"\n{message}"
    with assert_warns(DeprecationWarning, glob=glob):
        op(batch_size=2)


@contextmanager
def _record_legacy_op_kwargs(op_class):
    orig = op_class._legacy_op
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return orig(*args, **kwargs)

    op_class._legacy_op = staticmethod(spy)
    try:
        yield calls
    finally:
        op_class._legacy_op = orig


def test_capture_forwards_dynamic_api_name():
    # The deprecated test ops take no inputs, so they can't be captured - check what reaches the
    # legacy operator directly, both for the reader and for an operator in the captured graph.
    reader = ndd.readers.File(file_root=images_root)
    with _record_legacy_op_kwargs(ndd.readers.File) as reader_calls:
        with _record_legacy_op_kwargs(ndd._ops.Flip) as flip_calls:
            for jpegs, _ in reader.next_epoch(batch_size=2, capture=True):
                images = ndd.decoders.image(jpegs, device="cpu")
                flipped = ndd.flip(images, horizontal=1)
                assert _is_captured(flipped)
                ndd.as_tensor(flipped, pad=True)

    assert reader_calls and flip_calls
    for kwargs in reader_calls:
        assert kwargs.get("_module") == "nvidia.dali.experimental.dynamic.readers", kwargs
        assert kwargs.get("_display_name") == "File", kwargs
    for kwargs in flip_calls:
        assert kwargs.get("_module") == "nvidia.dali.experimental.dynamic", kwargs
        assert kwargs.get("_display_name") == "flip", kwargs
