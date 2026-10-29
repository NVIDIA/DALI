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
import tempfile

from nvidia.dali import fn, pipeline_def, types
from nose_utils import assert_raises


def run_reader(archive, index, dont_use_mmap, dtype=types.UINT8):
    @pipeline_def(batch_size=1, num_threads=1, device_id=None, prefetch_queue_depth=1)
    def pipeline():
        return fn.readers.webdataset(
            paths=[archive],
            index_paths=[index] if index else [],
            ext=["bin"],
            dtypes=[dtype],
            dont_use_mmap=dont_use_mmap,
            random_shuffle=False,
            initial_fill=1,
        )

    return pipeline().run()[0].as_array()


def test_invalid_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(1024))
        with open(index, "w") as f:
            f.write("v1.2 1\nbin 512 1024 sample.bin\n")
        with assert_raises(RuntimeError, glob="*requested range is outside*"):
            run_reader(archive, index, False)
