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
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import os

class RandomBBoxCropSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size,
                 boxes,
                 labels = None,
                 thresholds = [0, 0.01, 0.05, 0.1, 0.15],
                 scaling = [0.3, 1.0],
                 aspect_ratio = [0.5, 2.0],
                 ltrb = True,
                 num_attempts = 100,
                 allow_no_crop = False,
                 input_shape = None,
                 crop_shape = None,
                 num_threads=1, device_id=0, num_gpus=1):
        super(RandomBBoxCropSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.boxes = boxes
        self.labels = labels
        self.boxes_inputs = ops.ExternalSource()
        self.labels_inputs = ops.ExternalSource()

        self.bbox_crop = ops.RandomBBoxCrop(
            device = self.device,
            aspect_ratio = aspect_ratio,
            scaling = scaling,
            thresholds = thresholds,
            ltrb = ltrb,
            num_attempts = num_attempts,
            allow_no_crop = allow_no_crop,
            input_shape = input_shape,
            crop_shape = crop_shape)

    def define_graph(self):
        self.boxes_data = self.boxes_inputs()
        self.labels_data = self.labels_inputs()
        inputs = [self.boxes_data]
        if self.labels_data is not None:
            inputs.append(self.labels_data)
        return self.bbox_crop(*inputs)

    def iter_setup(self):
        self.feed_input(self.boxes_data, self.boxes)
        if self.labels is not None:
            self.feed_input(self.labels_data, self.labels)

def compare_eps(in1, in2, eps = 1e-6):
    diff1 = np.abs(in1 - in2)
    diff2 = np.abs(in2 - in1)
    err = np.mean( np.minimum(diff2, diff1) )
    return err < eps

def lt_eps(a, b, eps = 1e-6):
    return a + eps < b

def gt_eps(a, b, eps = 1e-6):
    return a - eps > b

def crop_contains(crop_anchor, crop_shape, point):
    ndim = len(crop_shape)
    assert(len(crop_shape) == ndim)
    assert(len(point) == ndim)
    for d in range(ndim):
        if lt_eps(point[d], crop_anchor[d]) or gt_eps(point[d], (crop_anchor[d] + crop_shape[d])):
            return False
    return True

def filter_by_centroid(crop_anchor, crop_shape, bboxes):
    ndim = len(crop_shape)
    nboxes = bboxes.shape[0]
    indexes = []
    for i in range(nboxes):
        bbox = bboxes[i]
        centroid = [0.5 * (bbox[d] + bbox[ndim + d]) for d in range(ndim)]
        if crop_contains(crop_anchor, crop_shape, centroid):
            indexes.append(i)
        filtered_boxes = np.array(bboxes[indexes, :])
    return filtered_boxes

def map_box(bbox, crop_anchor, crop_shape):
    ndim = int(len(bbox) / 2)
    assert len(crop_anchor) == ndim
    assert len(crop_shape) == ndim
    new_bbox = np.array(bbox)
    for d in range(ndim):
        c_start = crop_anchor[d]
        c_end = crop_anchor[d] + crop_shape[d]
        b_start = bbox[d]
        b_end = bbox[ndim+d]
        rel_extent = c_end - c_start
        n_start = (max(c_start, b_start) - c_start) / rel_extent
        n_end = (min(c_end, b_end) - c_start) / rel_extent
        new_bbox[d] = max(0.0, min(1.0, n_start))
        new_bbox[ndim + d] = max(0.0, min(1.0, n_end))
    return new_bbox

def check_processed_bboxes(crop_anchor, crop_shape, original_boxes, processed_boxes):
    filtered_boxes = filter_by_centroid(crop_anchor, crop_shape, original_boxes)
    assert(len(original_boxes) >= len(filtered_boxes))
    assert(len(filtered_boxes) == len(processed_boxes))
    nboxes = len(filtered_boxes)
    for i in range(nboxes):
        box = filtered_boxes[i]
        processed_box = processed_boxes[i]
        expected_box = map_box(box, crop_anchor, crop_shape)
        assert(compare_eps(expected_box, processed_box))

def check_crop_dims_variable_size(anchor, shape, scaling, aspect_ratio):
    ndim = len(shape)
    for d in range(ndim):
        assert(anchor[d] >= 0.0 and anchor[d] <= 1.0)
        assert(anchor[d] + shape[d] > 0.0 and anchor[d] + shape[d] <= 1.0)
        assert(shape[d] >= scaling[0] and shape[d] <= scaling[1])
        ar = shape[0] / shape[1]
        assert(ar >= aspect_ratio[0] and ar <= aspect_ratio[1])

def check_crop_dims_fixed_size(anchor, shape, expected_crop_shape, input_shape):
    ndim = len(shape)
    for d in range(ndim):
        assert(anchor[d] >= 0.0 and anchor[d] <= input_shape[d])
        assert shape[d] == expected_crop_shape[d], "{} != {}".format(shape, expected_crop_shape)
        assert(anchor[d] + shape[d] > 0.0 and anchor[d] + shape[d] <= input_shape[d])

bbox_2d_ltrb_1 = [0.0, 0.0, 0.9, 0.9]
bbox_2d_ltrb_2 = [0.1, 0.1, 0.99, 0.99]
bbox_2d_ltrb_3 = [0.3, 0.3, 0.5, 0.5]
bbox_3d_ltrb_1 = [0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
bbox_3d_ltrb_2 = [0.1, 0.1, 0.1, 0.6, 0.6, 0.6]
bbox_3d_ltrb_3 = [0.4, 0.4, 0.4, 0.9, 0.9, 0.9]

def get_bounding_boxes(ndim, nboxes, produce_labels=True):
    assert(nboxes == 3)
    if ndim == 2:
        boxes = [np.array([bbox_2d_ltrb_1, bbox_2d_ltrb_2, bbox_2d_ltrb_3], dtype=np.float32)]
    elif ndim == 3:
        boxes = [np.array([bbox_3d_ltrb_1, bbox_3d_ltrb_2, bbox_3d_ltrb_3], dtype=np.float32)]
    else:
        assert(False)
    return boxes

def get_labels(nboxes):
    assert(nboxes == 3)
    labels = [np.array([1, 2, 3], dtype=np.int32)]
    return labels

def check_random_bbox_crop_variable_shape(ndim, scaling, aspect_ratio):
    boxes = get_bounding_boxes(ndim, nboxes=3)
    labels = get_labels(nboxes=3)

    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=1,
                                           boxes=boxes, labels=labels,
                                           scaling=scaling, aspect_ratio=aspect_ratio,
                                           input_shape=None, crop_shape=None)
    pipe.build()
    for i in range(100):
        outputs = pipe.run()
        out_crop_anchor = outputs[0].at(0)
        out_crop_shape = outputs[1].at(0)
        out_boxes = outputs[2].at(0)
        out_labels = outputs[3].at(0)
        check_crop_dims_variable_size(out_crop_anchor, out_crop_shape, scaling, aspect_ratio)
        check_processed_bboxes(out_crop_anchor, out_crop_shape, boxes[0], out_boxes)


def test_random_bbox_crop_variable_shape():
    for ndim in [2, 3]:
        for scaling in [[0.3, 0.5], [0.1, 0.3], [0.9, 0.99]]:
            for aspect_ratio in [[0.01, 100], [0.5, 2.0]]:
                yield check_random_bbox_crop_variable_shape, \
                        ndim, scaling, aspect_ratio

def check_random_bbox_crop_fixed_shape(ndim, crop_shape, input_shape):
    boxes = get_bounding_boxes(ndim, nboxes=3)
    labels = get_labels(nboxes=3)

    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=1,
                                           boxes=boxes, labels=labels,
                                           scaling=None, aspect_ratio=None,
                                           input_shape=input_shape, crop_shape=crop_shape)
    pipe.build()
    for i in range(100):
        outputs = pipe.run()
        out_crop_anchor = outputs[0].at(0)
        out_crop_shape = outputs[1].at(0)
        out_boxes = outputs[2].at(0)
        out_labels = outputs[3].at(0)
        check_crop_dims_fixed_size(out_crop_anchor, out_crop_shape, crop_shape, input_shape)
        rel_out_crop_anchor = [out_crop_anchor[d] / input_shape[d] for d in range(ndim)]
        rel_out_crop_shape = [out_crop_shape[d] / input_shape[d] for d in range(ndim)]
        check_processed_bboxes(rel_out_crop_anchor, rel_out_crop_shape, boxes[0], out_boxes)


def test_random_bbox_crop_fixed_shape():
    input_shapes = {
        2: [[400, 300]],
        3: [[400, 300, 64]]
    }

    crop_shapes = {
        2: [[100, 50], [400, 300]],
        3: [[100, 50, 32], [400, 300, 64]]
    }
    for ndim in [2, 3]:
        for input_shape in input_shapes[ndim]:
            for crop_shape in crop_shapes[ndim]:
                yield check_random_bbox_crop_fixed_shape, \
                        ndim, crop_shape, input_shape

def main():
  for test in test_random_bbox_crop_fixed_shape():
      test[0](*test[1:])

  for test in test_random_bbox_crop_variable_shape():
      test[0](*test[1:])

if __name__ == '__main__':
  main()
