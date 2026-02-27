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

from ndd_vs_fn_test_utils import sign_off
from nvidia.dali.experimental.dynamic._ops import _all_ops


excluded_operators = [
    "readers.VideoResize",  # TODO(michalz): add manual tests
    "readers.TFRecord",  # TODO(michalz): add tests
    "experimental.readers.Fits",  # TODO(michalz): add tests
    "roi_random_crop",  # TODO(michalz): add tests
    "plugin.video.decoder",
]


def get_all_operators():
    ret = []
    for o in _all_ops:
        if o._schema.IsInternal() or o._schema.IsDocHidden() or o._schema_name.startswith("_"):
            continue  # skip internal/hidden operators
        ret.append(o._op_path if o._is_reader else o._fn_path)
    return ret


def test_coverage():
    covered_operators = sign_off.tested_ops

    failure = ""
    excluded_but_tested = covered_operators.intersection(excluded_operators)
    if excluded_but_tested:
        print("\n\nThe following operators are marked as excluded but were covered in the tests:")
        for op in excluded_but_tested:
            print(f"    {op}")
        failure += f"The test covers {len(excluded_but_tested)} excluded operator(s).\n"

    eligible_operators = set(get_all_operators()).difference(excluded_operators)

    untested_operators = [op for op in eligible_operators if op not in covered_operators]

    if untested_operators:
        print("\n\nTest coverage gap detected!\n\nOperators that are not covered:")
        for op in sorted(untested_operators):
            print(f"  - {op}")
        print(f"\nTotal not covered: {len(untested_operators)} out of {len(eligible_operators)}")
        failure += (
            f"Found {len(untested_operators)} operator(s) that were neither tested nor excluded.\n"
        )
        if len(excluded_operators):
            print(f"{len(excluded_operators)} operator(s) were excluded from the test.")
    else:
        if len(excluded_operators):
            print("All eligible operators are tested.")
            print(f"{len(excluded_operators)} operator(s) were excluded from the test.")
        else:
            print("All operators are tested.")

    assert not failure, failure
