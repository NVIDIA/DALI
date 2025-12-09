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


def test_ndd_operator_api():
    for op in ndd._ops._all_functions:
        for member in dir(op):
            assert member.startswith("_"), f"Unexpected public member `{member}` in {op}"


def test_ndd_reader_api():
    for reader in dir(ndd.readers):
        reader_cls = getattr(ndd.readers, reader)
        if isinstance(reader_cls, type) and issubclass(reader_cls, ndd._ops.Reader):
            allowed_members = ["next_epoch"]
            for member in dir(reader_cls):
                assert (
                    member.startswith("_") or member in allowed_members
                ), f"Unexpected public member `{member}` in {reader}"
