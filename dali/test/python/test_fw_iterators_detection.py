# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import time
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator


DALI_EXTRA_PATH = os.environ['DALI_EXTRA_PATH']
EPOCH_SIZE = 32
BATCH_SIZE = 1

class DetectionPipeline(Pipeline):
    def __init__(self, batch_size, device_id, file_root, annotations_file):
        super(DetectionPipeline, self).__init__(
            batch_size, 2, device_id, True, 12)

        # Reading COCO dataset
        self.input = ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=device_id,
            num_shards=1,
            ratio=True,
            ltrb=True)

    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        return inputs, boxes.gpu(), labels

def data_paths():
    if not DALI_EXTRA_PATH:
        coco = '/data/coco/coco-2017/coco2017/'
        root = os.path.join(coco, 'val2017')
        annotations = os.path.join(
            coco, 'annotations/instances_val2017.json')
    else:
        root = os.path.join(DALI_EXTRA_PATH, 'db', 'coco', 'images')
        annotations = os.path.join(DALI_EXTRA_PATH, 'db', 'coco', 'instances.json')

    return root, annotations

#####################################
########## Unit tests ###############
#####################################

def test_mxnet_pipeline_dynamic_shape():
    root, annotations = data_paths()
    pipeline = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = MXNetIterator([pipeline], [('data', MXNetIterator.DATA_TAG),
                                              ('bboxes', MXNetIterator.LABEL_TAG),
                                              ('label', MXNetIterator.LABEL_TAG)],
                                              EPOCH_SIZE, auto_reset=False,
                                              dynamic_shape=True)
    for data in train_loader:
        assert data is not None


def test_pytorch_pipeline_dynamic_shape():
    root, annotations = data_paths()
    pipeline = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = PyTorchIterator([pipeline], ['data', 'bboxes', 'label'],
                                   EPOCH_SIZE, auto_reset=False,
                                   dynamic_shape=True)
    for data in train_loader:
        assert data is not None
