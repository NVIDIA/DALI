# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.fn as fn
import os
import random
from nvidia.dali import pipeline_def
import test_utils
from nose2.tools import params
from nose_utils import assert_raises

data_path = os.path.join(test_utils.get_dali_extra_path(), "db/single/jpeg/")
batch_size = 3

rel_fpaths = [
    "db/single/jpeg/450/bobsled-683993_1280.jpg",
    "db/single/jpeg/450/bobsled-643397_1280.jpg",
    "db/single/jpeg/450/bobsled-683995_1280.jpg",
    "db/single/jpeg/312/grasshopper-4357907_1280.jpg",
    "db/single/jpeg/312/grasshopper-4357903_1280.jpg",
    "db/single/jpeg/312/cricket-1345065_1280.jpg",
    "db/single/jpeg/372/baboon-3089012_1280.jpg",
    "db/single/jpeg/372/monkey-653705_1920.jpg",
    "db/single/jpeg/372/baboon-174073_1280.jpg",
    "db/single/jpeg/721/pillows-1031079_1280.jpg",
    "db/single/jpeg/721/pillows-820149_1280.jpg",
    "db/single/jpeg/721/pillow-2071096_1280.jpg",
    "db/single/jpeg/425/abandoned-2179173_1280.jpg",
    "db/single/jpeg/425/barn-227557_1280.jpg",
    "db/single/jpeg/425/winter-barn-556696_1280.jpg",
]
full_paths = [os.path.join(test_utils.get_dali_extra_path(), rel_path) for rel_path in rel_fpaths]


def get_filepaths(dtype):
    path_strs = [random.choice(full_paths) for _ in range(batch_size)]
    paths_output = [np.frombuffer(path_str.encode("utf-8"), dtype=dtype) for path_str in path_strs]
    data_output = [np.fromfile(path, dtype=np.uint8) for path in path_strs]
    return paths_output, data_output


@params(np.int8, np.uint8)
def test_io_file_read(dtype):
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipe():
        fpath, data_ref = fn.external_source(source=lambda: get_filepaths(dtype), num_outputs=2)
        data = fn.io.file.read(fpath)
        return data, data_ref

    p = pipe()

    for _ in range(4):
        data, data_ref = p.run()
        for i in range(batch_size):
            np.testing.assert_array_equal(data[i], data_ref[i])


def test_io_file_const_str_err():
    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipe():
        data = fn.io.file.read(full_paths[:batch_size])
        return data

    with assert_raises(
        TypeError,
        glob="expected inputs of type 'DataNode', list of 'DataNode' "
        "or convertible to constant nodes.",
    ):
        p = pipe()
        p.build()
