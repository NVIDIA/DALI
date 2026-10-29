# Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import struct
import tempfile

import numpy as np

from nvidia.dali import fn, pipeline_def
from nose2.tools import cartesian_params
from nose_utils import assert_raises


@cartesian_params((False, True), (False, True), ((1 << 61) - 16, 9))
def test_numpy_reader_rejects_payload_outside_file(
    dont_use_mmap, cache_header_information, extent
):
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "invalid.npy")
        header = repr({"descr": "<f8", "fortran_order": False, "shape": (extent,)})
        header = header.ljust(117) + "\n"
        with open(path, "wb") as f:
            f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)))
            f.write(header.encode())
            f.write(bytes(64))

        @pipeline_def(batch_size=1, device_id=None, num_threads=1)
        def pipe():
            return fn.readers.numpy(
                files=[path],
                dont_use_mmap=dont_use_mmap,
                cache_header_information=cache_header_information,
            )

        assert_raises(
            RuntimeError, pipe().run, glob="*payload exceeds the available file data*"
        )


@cartesian_params((False, True), (False, True))
def test_numpy_reader_valid_payload(dont_use_mmap, cache_header_information):
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "valid.npy")
        expected = np.arange(8, dtype=np.float64)
        np.save(path, expected)

        @pipeline_def(batch_size=1, device_id=None, num_threads=1)
        def pipe():
            return fn.readers.numpy(
                files=[path],
                dont_use_mmap=dont_use_mmap,
                cache_header_information=cache_header_information,
            )

        p = pipe()
        for _ in range(2):
            np.testing.assert_array_equal(p.run()[0].at(0), expected)
