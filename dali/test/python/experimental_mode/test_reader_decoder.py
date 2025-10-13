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

import nvidia.dali.experimental.dynamic as D
from nose2.tools import params
import os
from test_utils import get_dali_extra_path


dali_extra_path = get_dali_extra_path()


@params("cpu", "gpu")
def test_reader_batch(device_type):
    reader = D.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
    )

    iters = 0
    for file, lbl in reader.next_epoch(batch_size=4):
        assert isinstance(file, D.Batch)
        assert isinstance(lbl, D.Batch)
        assert file.batch_size == 4
        assert lbl.batch_size == 4
        assert file.dtype == D.uint8
        assert lbl.dtype == D.int32
        assert file.device == D.Device("cpu")
        assert lbl.device == D.Device("cpu")
        file.evaluate()
        img = D.decoders.image(file, device=device_type)
        img.evaluate()
        assert img.dtype == D.uint8
        assert len(img.shape[0]) == 3  # HWC
        assert img.shape[0][2] == 3  # RGB
        assert img.device == D.Device("cpu" if device_type == "cpu" else "gpu")
        iters += 1
    assert iters > 0


@params("cpu", "gpu")
def test_reader_sample(device_type):
    reader = D.readers.File(
        file_root=os.path.join(dali_extra_path, "db", "single", "jpeg"),
        file_list=os.path.join(dali_extra_path, "db", "single", "jpeg", "image_list.txt"),
    )

    iters = 0
    for file, lbl in reader.next_epoch(batch_size=None):
        assert isinstance(file, D.Tensor)
        assert isinstance(lbl, D.Tensor)
        assert file.dtype == D.uint8
        assert lbl.dtype == D.int32
        assert file.device == D.Device("cpu")
        assert lbl.device == D.Device("cpu")
        file.evaluate()
        img = D.decoders.image(file, device=device_type)
        img.evaluate()
        assert img.dtype == D.uint8
        assert img.shape[2] == 3  # RGB
        assert img.device == D.Device("cpu" if device_type == "cpu" else "gpu")
        iters += 1
    assert iters > 0
