# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from test_utils import get_dali_extra_path

import numpy as np


data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
batch_size = 16


def create_py_file_reader(images_dir):
    with open(os.path.join(images_dir, "image_list.txt"), 'r') as f:
        file_label = [line.rstrip().split(' ') for line in f if line != '']
        files, labels = zip(*file_label)

    def py_file_reader(sample_info):
        sample_idx = sample_info.idx_in_epoch
        jpeg_filename = files[sample_idx]
        label = np.int32([labels[sample_idx]])
        with open(os.path.join(images_dir, jpeg_filename), 'rb') as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded_img, label

    return py_file_reader


@pipeline_def(py_start_method='fork', batch_size=batch_size)
def simple_pipeline():
    jpegs, labels = fn.external_source(source=create_py_file_reader(images_dir), num_outputs=2, parallel=True, batch=False)
    images = fn.decoders.image(jpegs, device="cpu")
    return images, labels


def test_no_segfault():
    """
    This may cause segmentation fault on Python teardown if shared memory wrappers managed by the py_pool
    are garbage collected before pipeline's backend
    """
    pipe = simple_pipeline(batch_size=batch_size, py_start_method='fork', num_threads=4, prefetch_queue_depth=2, device_id=0)
    pipe.build()
    pipe.run()
