# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path
import os
from nose.tools import raises
import tempfile

test_data_root = get_dali_extra_path()
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
train_annotations = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

test_data = {
    'car-race-438467_1280.jpg' : 0,
    'clock-1274699_1280.jpg' : 5,
    'kite-1159538_1280.jpg' : 6,
    'cow-234835_1280.jpg' : 17,
    'home-office-336378_1280.jpg' : 21,
    'suit-2619784_1280.jpg' : 39,
    'business-suit-690048_1280.jpg' : 41,
    'car-604019_1280.jpg' : 59
}

files = list(test_data.keys())
expected_ids = list(test_data.values())

def test_operator_coco_reader():
    with tempfile.TemporaryDirectory() as annotations_dir:
        pipeline = Pipeline(batch_size=2, num_threads=4, device_id=0)
        with pipeline:
            inputs, _, _, ids = fn.coco_reader(
                file_root=file_root,
                annotations_file=train_annotations,
                image_ids=True,
                files=files,
                save_preprocessed_annotations=True,
                save_preprocessed_annotations_dir=annotations_dir)
            pipeline.set_outputs(ids)
        pipeline.build()

        i = 0
        while i < len(files):
            out = pipeline.run()
            assert out[0].at(0) == expected_ids[i]
            assert out[0].at(1) == expected_ids[i + 1]
            i = i + 2

        filenames_file = os.path.join(annotations_dir, 'filenames.dat')
        with open(filenames_file) as f:
            lines = f.read().splitlines()
        assert lines.sort() == files.sort()


@raises(RuntimeError)
def test_invalid_args():
    pipeline = Pipeline(batch_size=2, num_threads=4, device_id=0)
    with pipeline:
        inputs, _, _, ids = fn.coco_reader(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            files=files,
            preprocessed_annotations_dir='/tmp')
        pipeline.set_outputs(ids)
    pipeline.build()

