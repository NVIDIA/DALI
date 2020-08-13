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
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import os

bbox_2d_ltrb_1 = [0.0123, 0.0123, 0.2123, 0.2123]
bbox_2d_ltrb_2 = [0.1123, 0.1123, 0.19123, 0.19123]
bbox_2d_ltrb_3 = [0.3123, 0.3123, 0.5123, 0.5123]
bbox_3d_ltrb_1 = [0.123, 0.6123, 0.6123, 0.7123, 0.7123, 0.7123]
bbox_3d_ltrb_2 = [0.1123, 0.1123, 0.1123, 0.2123, 0.2123, 0.2123]
bbox_3d_ltrb_3 = [0.7123, 0.7123, 0.7123, 0.8123, 0.8123, 0.8123]

bboxes_data = {
    2 : [bbox_2d_ltrb_1, bbox_2d_ltrb_2, bbox_2d_ltrb_3],
    3 : [bbox_3d_ltrb_1, bbox_3d_ltrb_2, bbox_3d_ltrb_3]
}

class BBoxDataIterator():
    def __init__(self, n, batch_size, ndim = 2, produce_labels = False):
        self.batch_size = batch_size
        self.ndim = ndim
        self.produce_labels = produce_labels
        self.num_outputs = 2 if produce_labels else 1
        self.n = n
        self.i = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        # return a copy, so that the iteration number doesn't collide
        return BBoxDataIterator(self.n, self.batch_size, self.ndim, self.produce_labels)

    def __next__(self):
        boxes = []
        labels = []
        bboxes = bboxes_data[self.ndim]
        if self.i % 2 == 0:
            boxes.append(np.array([bboxes[0], bboxes[1], bboxes[2]], dtype=np.float32))
            labels.append(np.array([1, 2, 3], dtype=np.int32))
            if self.batch_size > 1:
                boxes.append(np.array([bboxes[2], bboxes[1]], dtype=np.float32))
                labels.append(np.array([2, 1], dtype=np.int32))
                for i in range(self.batch_size - 2):
                    boxes.append(np.array([bboxes[2]], dtype=np.float32))
                    labels.append(np.array([3], dtype=np.int32))
        else:
            boxes.append(np.array([bboxes[2]], dtype=np.float32))
            labels.append(np.array([3], dtype=np.int32))
            if self.batch_size > 1:
                boxes.append(np.array([bboxes[1], bboxes[2], bboxes[0]], dtype=np.float32))
                labels.append(np.array([2, 3, 1], dtype=np.int32))
                for i in range(self.batch_size - 2):
                    boxes.append(np.array([bboxes[1]], dtype=np.float32))
                    labels.append(np.array([2], dtype=np.int32))

        if self.i < self.n:
            self.i = self.i + 1
            outputs = [boxes]
            if self.produce_labels:
                outputs.append(labels)
            return outputs
        else:
            self.i = 0
            raise StopIteration
    next = __next__


class RandomBBoxCropSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size,
                 bbox_source,
                 thresholds = [0, 0.01, 0.05, 0.1, 0.15],
                 threshold_type = 'iou',
                 scaling = [0.3, 1.0],
                 aspect_ratio = [0.5, 2.0],
                 bbox_layout = "xyXY",
                 num_attempts = 100,
                 allow_no_crop = False,
                 input_shape = None,
                 crop_shape = None,
                 all_boxes_above_threshold = False,
                 num_threads=1, device_id=0, num_gpus=1):
        super(RandomBBoxCropSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)

        self.device = device
        self.bbox_source = bbox_source

        self.bbox_crop = ops.RandomBBoxCrop(
            device = self.device,
            aspect_ratio = aspect_ratio,
            scaling = scaling,
            thresholds = thresholds,
            threshold_type = threshold_type,
            bbox_layout = bbox_layout,
            num_attempts = num_attempts,
            allow_no_crop = allow_no_crop,
            input_shape = input_shape,
            crop_shape = crop_shape,
            all_boxes_above_threshold = all_boxes_above_threshold)

    def define_graph(self):
        inputs = fn.external_source(source=self.bbox_source, num_outputs=self.bbox_source.num_outputs)
        anchor, shape, boxes = self.bbox_crop(*inputs)

        sx = fn.slice(shape,
                      types.Constant([0.0], device='cpu'),
                      types.Constant([1.0], device='cpu'),
                      normalized_anchor=False, normalized_shape=False,
                      axes=[0])
        sy = fn.slice(shape,
                      types.Constant([1.0], device='cpu'),
                      types.Constant([1.0], device='cpu'),
                      normalized_anchor=False, normalized_shape=False,
                      axes=[0])

        mx = (1.0 / sx) * fn.constant(
            device='cpu', 
            fdata = (1.0, 0.0, 
                     0.0, 0.0)
        )
        my = (1.0 / sy) * fn.constant(
            device='cpu', 
            fdata = (0.0, 0.0, 
                     0.0, 1.0)
        )
        m = mx + my

        t = -1.0 * anchor

        return [inputs[0], anchor, shape, boxes, m, t]

def check_random_bbox_crop_variable_shape(batch_size, ndim, scaling, aspect_ratio, use_labels):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=batch_size,
                                           bbox_source=bbox_source,
                                           bbox_layout=bbox_layout,
                                           scaling=scaling, aspect_ratio=aspect_ratio,
                                           input_shape=None, crop_shape=None)
    pipe.build()
    for i in range(1):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_boxes = outputs[0].at(sample)
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            out_boxes = outputs[3].at(sample)
            m = outputs[4].at(sample)
            t = outputs[5].at(sample)

            print("anchor: ", out_crop_anchor)
            print("shape: ", out_crop_shape)
            print("M: ", m)
            print("T: ", t)

scaling = [0.3, 0.5]
aspect_ratio = [0.5, 2.0]
check_random_bbox_crop_variable_shape(1, 2, scaling, aspect_ratio, False)
