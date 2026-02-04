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

import test_ndd_vs_fn_1darray
import test_ndd_vs_fn_noinput
import test_ndd_vs_fn_image
import test_ndd_vs_fn_random
import test_ndd_vs_fn_sequence
import test_ndd_vs_fn_decoders
import test_ndd_vs_fn_numpy
import test_ndd_vs_fn_readers
import test_ndd_vs_fn_readers_nemo_asr
import test_ndd_vs_fn
from nvidia.dali.experimental.dynamic._ops import _all_ops


excluded_operators = [
    "_arithmetic_generic_op",  # Hidden operators are not part of this suite.
    "_conditional.merge",  # Hidden operators are not part of this suite.
    "_conditional.not_",  # Hidden operators are not part of this suite.
    "_conditional.split",  # Hidden operators are not part of this suite.
    "_conditional.validate_logical",  # Hidden operators are not part of this suite.
    "_shape",  # Hidden operators are not part of this suite.
    "_subscript_dim_check",  # Hidden operators are not part of this suite.
    "_tensor_subscript",  # Hidden operators are not part of this suite.
    "batch_permutation",  # BUG
    "bbox_rotate",  # BUG
    "cat",  # BUG
    "decoders.image_random_crop",  # BUG
    "decoders.image_slice",  # BUG
    "decoders.inflate",  # TODO(mszolucha): Add inflate test.
    "experimental.decoders.image_random_crop",  # BUG
    "experimental.inputs.video",  # Is it there?
    "experimental.readers.fits",  # No input data in DALI_extra
    "io.file.read",  # BUG
    "permute_batch",  # BUG
    "random_resized_crop",  # BUG
    "readers.numpy",  # How to use GDS here?
    "readers.video_resize",  # BUG
    "slice",  # BUG
    "stack",  # BUG
    "warp_affine",  # BUG
]


def get_tested_operators():
    tested_operators = set()
    modules = [
        test_ndd_vs_fn_1darray,
        test_ndd_vs_fn_noinput,
        test_ndd_vs_fn_image,
        test_ndd_vs_fn_random,
        test_ndd_vs_fn_sequence,
        test_ndd_vs_fn_decoders,
        test_ndd_vs_fn_numpy,
        test_ndd_vs_fn_readers,
        test_ndd_vs_fn_readers_nemo_asr,
        test_ndd_vs_fn,
    ]
    for module in modules:
        if hasattr(module, "tested_operators"):
            tested_operators.update(module.tested_operators)
    return tested_operators


def get_all_operators():
    ret = []
    for o in _all_ops:
        op_name = o._schema.ModulePath()
        op_name.append(o._fn_name)
        ret.append(".".join(op_name))
    return ret


def test_coverage():
    tested_operators = get_tested_operators()
    covered_operators = tested_operators.union(excluded_operators)
    all_operators = get_all_operators()

    untested_operators = [op for op in all_operators if op not in covered_operators]

    if untested_operators:
        print("\nOperators that are not covered:")
        for op in sorted(untested_operators):
            print(f"  - {op}")
        print(f"\nTotal not covered: {len(untested_operators)} out of {len(all_operators)}")
    else:
        print("All operators are tested!")

    assert len(untested_operators) == 0, f"Found {len(untested_operators)} untested operators"
