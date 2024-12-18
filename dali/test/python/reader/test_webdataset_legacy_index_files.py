# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def, fn
import os
import glob
from test_utils import get_dali_extra_path

test_data_root = os.path.join(get_dali_extra_path(), "db", "webdataset", "legacy_index_formats")


@pipeline_def(batch_size=8, num_threads=4, device_id=0)
def wds_index_file_pipeline(idx_path, device):
    jpg, cls = fn.readers.webdataset(
        paths=[os.path.join(test_data_root, "data.tar")], index_paths=[idx_path], ext=["jpg", "cls"]
    )
    if device == "gpu":
        jpg = jpg.gpu()
        cls = cls.gpu()
    return jpg, cls


def _test_wds_index_file_pipeline(idx_path, device):
    p = wds_index_file_pipeline(idx_path, device)
    p.run()


def test_wds_index_file_pipeline():
    idx_files = glob.glob(test_data_root + "/*.idx")
    for idx_path in idx_files:
        for device in ["cpu", "gpu"]:
            yield _test_wds_index_file_pipeline, idx_path, device
