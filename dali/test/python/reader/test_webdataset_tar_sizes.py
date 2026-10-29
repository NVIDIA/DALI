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

import io
import os
import tarfile
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


@params(*[(mmap, size) for mmap in (False, True) for size in (-4, 8**11 - 1, 2048)])
def test_invalid_tar_entry_size(dont_use_mmap, size):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        header = bytearray(tarfile.TarInfo("sample.bin").tobuf())
        field = b"-4" if size == -4 else oct(size)[2:].encode()
        header[124:136] = field.rjust(11, b" ") + b"\0"
        header[148:156] = b" " * 8
        header[148:156] = f"{sum(header):06o}\0 ".encode()
        with open(archive, "wb") as f:
            f.write(header + bytes(1024))
        with assert_raises(RuntimeError, glob="*extends beyond the archive*"):
            run_reader(archive, None, dont_use_mmap, types.FLOAT)


@params(False, True)
def test_valid_tar(dont_use_mmap):
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "data.tar")
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("sample.bin")
            info.size = 4
            tar.addfile(info, io.BytesIO(bytes(range(4))))
        assert run_reader(archive, None, dont_use_mmap).tolist() == [[0, 1, 2, 3]]
