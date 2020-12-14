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

test_data_root = get_dali_extra_path()
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
train_annotations = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

files = [
    'car-race-438467_1280.jpg', 
    'clock-1274699_1280.jpg', 
    'kite-1159538_1280.jpg', 
    'cow-234835_1280.jpg', 
    'home-office-336378_1280.jpg', 
    'suit-2619784_1280.jpg', 
    'business-suit-690048_1280.jpg', 
    'car-604019_1280.jpg']


def get_pipeline():
    pipeline = Pipeline(batch_size=2, num_threads=4, device_id=0)
    
    with pipeline:
        inputs, _, _, ids = fn.coco_reader(
            file_root=file_root,
            annotations_file=train_annotations,
            image_ids=True,
            files=files,
            save_preprocessed_annotations=True,
            save_preprocessed_annotations_dir='/tmp')
        pipeline.set_outputs(ids)

    return pipeline


def test_operator_coco_reader():
    pipeline = get_pipeline()
    pipeline.build()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [0, 5]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [6, 17]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [21, 39]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [41, 59]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [0, 5]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [6, 17]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [21, 39]).all()

    out = pipeline.run()
    assert (out[0].as_array().ravel() == [41, 59]).all()


    with open('/tmp/filenames.dat') as f:
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

