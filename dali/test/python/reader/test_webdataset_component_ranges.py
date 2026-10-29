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
from nose2.tools import params
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


@params(
    *[
        (mmap, offset, size)
        for mmap in (False, True)
        for offset, size in (
            (512, 2**64 - 4),
            (512, 2**64 - 512),
            (-512, 4),
            (512, 1028),
            (2048, 4),
        )
    ]
)
def test_invalid_component_range(dont_use_mmap, offset, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(1024))
        with open(index, "w") as f:
            f.write(f"v1.2 1\nbin {offset} {size} sample.bin\n")
        with assert_raises(RuntimeError, glob="*offset or size is outside*"):
            run_reader(archive, index, dont_use_mmap, types.FLOAT)


@params(False, True)
def test_component_ending_at_eof(dont_use_mmap):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        index = os.path.join(tmp, "data.idx")
        with open(archive, "wb") as f:
            f.write(bytes(512) + bytes(range(4)))
        with open(index, "w") as f:
            f.write("v1.2 1\nbin 512 4 sample.bin\n")
        assert run_reader(archive, index, dont_use_mmap).tolist() == [[0, 1, 2, 3]]
