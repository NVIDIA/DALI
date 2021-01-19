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
import random

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
                for _ in range(self.batch_size - 2):
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
                 output_bbox_indices = False,
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
            all_boxes_above_threshold = all_boxes_above_threshold,
            output_bbox_indices = output_bbox_indices
        )

    def define_graph(self):
        inputs = fn.external_source(source=self.bbox_source, num_outputs=self.bbox_source.num_outputs)
        outputs = self.bbox_crop(*inputs)
        return [inputs[0], *outputs]

def crop_contains(crop_anchor, crop_shape, point):
    ndim = len(crop_shape)
    assert(len(crop_shape) == ndim)
    assert(len(point) == ndim)

    point = np.array(point)
    crop_anchor = np.array(crop_anchor)
    crop_shape = np.array(crop_shape)

    if np.any(np.less(point, crop_anchor)) or np.any(np.greater(point, (crop_anchor + crop_shape))):
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

def check_processed_bboxes(crop_anchor, crop_shape, original_boxes, processed_boxes, bbox_indices=None):
    if bbox_indices is not None:
        filtered_boxes = np.array(original_boxes[bbox_indices])
    else:
        filtered_boxes = filter_by_centroid(crop_anchor, crop_shape, original_boxes)
    assert(len(original_boxes) >= len(filtered_boxes))
    assert(len(filtered_boxes) == len(processed_boxes))
    nboxes = len(filtered_boxes)
    for i in range(nboxes):
        box = filtered_boxes[i]
        processed_box = processed_boxes[i]
        expected_box = map_box(box, crop_anchor, crop_shape)
        assert(np.allclose(expected_box, processed_box, atol=1e-6))

def check_crop_dims_variable_size(anchor, shape, scaling, aspect_ratio):
    ndim = len(shape)
    k = 0
    nranges = len(aspect_ratio) / 2

    max_extent = 0.0
    for d in range(ndim):
        max_extent = shape[d] if shape[d] > max_extent else max_extent
    assert (max_extent >= scaling[0] or np.isclose(max_extent, scaling[0]))
    assert (max_extent <= scaling[1] or np.isclose(max_extent, scaling[1]))
    for d in range(ndim):
        assert anchor[d] >= 0.0 and anchor[d] <= 1.0, anchor
        assert(anchor[d] + shape[d] > 0.0 and anchor[d] + shape[d] <= 1.0)

        for d2 in range(d+1, ndim):
            ar = shape[d] / shape[d2]
            ar_min = aspect_ratio[k*2]
            ar_max = aspect_ratio[k*2+1]
            if ar_min == ar_max:
                assert np.isclose(ar, ar_min), "ar {}/{} = {} is not close to ar_min={}".format(d, d2, ar, ar_min)
            else:
                assert ar >= aspect_ratio[k*2] and ar <= aspect_ratio[k*2+1]
            k = int((k + 1) % nranges)

def check_crop_dims_fixed_size(anchor, shape, expected_crop_shape, input_shape):
    ndim = len(shape)
    for d in range(ndim):
        assert(anchor[d] >= 0.0 and anchor[d] <= input_shape[d])
        assert shape[d] == expected_crop_shape[d], "{} != {}".format(shape, expected_crop_shape)
        assert(anchor[d] + shape[d] > 0.0)

def check_random_bbox_crop_variable_shape(batch_size, ndim, scaling, aspect_ratio, use_labels, output_bbox_indices):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=batch_size,
                                           bbox_source=bbox_source,
                                           bbox_layout=bbox_layout,
                                           scaling=scaling, aspect_ratio=aspect_ratio,
                                           input_shape=None, crop_shape=None,
                                           output_bbox_indices=output_bbox_indices)
    pipe.build()
    for i in range(100):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_boxes = outputs[0].at(sample)
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            out_boxes = outputs[3].at(sample)
            check_crop_dims_variable_size(out_crop_anchor, out_crop_shape, scaling, aspect_ratio)
            bbox_indices_out_idx = 4 if not use_labels else 5
            bbox_indices = outputs[bbox_indices_out_idx].at(sample) if output_bbox_indices else None
            check_processed_bboxes(out_crop_anchor, out_crop_shape, in_boxes, out_boxes, bbox_indices)


def test_random_bbox_crop_variable_shape():
    random.seed(1234)
    for batch_size in [3]:
        for ndim in [2, 3]:
            for scaling in [[0.3, 0.5], [0.1, 0.3], [0.9, 0.99]]:
                aspect_ratio_ranges = {
                    2 :  [[0.01, 100], [0.5, 2.0], [1.0, 1.0]],
                    3 :  [[0.5, 2.0, 0.6, 2.1, 0.4, 1.9], [1.0, 1.0], [0.5, 0.5, 0.25, 0.25, 0.5, 0.5]]
                }
                for aspect_ratio in aspect_ratio_ranges[ndim]:
                    use_labels = random.choice([True, False])
                    out_bbox_indices = random.choice([True, False])
                    yield check_random_bbox_crop_variable_shape, \
                        batch_size, ndim, scaling, aspect_ratio, use_labels, out_bbox_indices

def check_random_bbox_crop_fixed_shape(batch_size, ndim, crop_shape, input_shape, use_labels):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=batch_size,
                                           bbox_source=bbox_source,
                                           bbox_layout=bbox_layout,
                                           scaling=None, aspect_ratio=None,
                                           input_shape=input_shape, crop_shape=crop_shape,
                                           all_boxes_above_threshold = False)
    pipe.build()
    for i in range(100):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_boxes = outputs[0].at(sample)
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            out_boxes = outputs[3].at(sample)
            check_crop_dims_fixed_size(out_crop_anchor, out_crop_shape, crop_shape, input_shape)
            rel_out_crop_anchor = [out_crop_anchor[d] / input_shape[d] for d in range(ndim)]
            rel_out_crop_shape = [out_crop_shape[d] / input_shape[d] for d in range(ndim)]
            check_processed_bboxes(rel_out_crop_anchor, rel_out_crop_shape, in_boxes, out_boxes)


def test_random_bbox_crop_fixed_shape():
    input_shapes = {
        2: [[400, 300]],
        3: [[400, 300, 64]]
    }

    crop_shapes = {
        2: [[100, 50], [400, 300], [600, 400]],
        3: [[100, 50, 32], [400, 300, 64], [600, 400, 48]]
    }
    for batch_size in [3]:
        for ndim in [2, 3]:
            for input_shape in input_shapes[ndim]:
                for crop_shape in crop_shapes[ndim]:
                    for use_labels in [True, False]:
                        yield check_random_bbox_crop_fixed_shape, \
                                batch_size, ndim, crop_shape, input_shape, use_labels

def check_random_bbox_crop_overlap(batch_size, ndim, crop_shape, input_shape, use_labels):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=batch_size,
                                           thresholds=[1.0],
                                           threshold_type='overlap',
                                           num_attempts=1000,
                                           bbox_source=bbox_source,
                                           bbox_layout=bbox_layout,
                                           scaling=None, aspect_ratio=None,
                                           input_shape=input_shape, crop_shape=crop_shape,
                                           all_boxes_above_threshold = False)
    pipe.build()
    for _ in range(100):
        outputs = pipe.run()
        for sample in range(batch_size):
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            rel_out_crop_anchor = [out_crop_anchor[d] / input_shape[d] for d in range(ndim)]
            rel_out_crop_shape = [out_crop_shape[d] / input_shape[d] for d in range(ndim)]
            in_boxes = outputs[0].at(sample)
            nboxes = in_boxes.shape[0]
            at_least_one_box_in = False
            for box_idx in range(nboxes):
                box = in_boxes[box_idx]
                is_box_in = True
                for d in range(ndim):
                    if rel_out_crop_anchor[d] > box[d] or \
                        (rel_out_crop_anchor[d] + rel_out_crop_shape[d]) < box[ndim + d]:
                        is_box_in = False
                        break

                if is_box_in:
                    at_least_one_box_in = True
                    break
            assert(at_least_one_box_in)

def test_random_bbox_crop_overlap():
    input_shapes = {
        2: [[400, 300]],
        3: [[400, 300, 64]]
    }
    crop_shapes = {
        2: [[150, 150], [400, 300]],
        3: [[50, 50, 32], [400, 300, 64]]
    }
    for batch_size in [3]:
        for ndim in [2, 3]:
            for input_shape in input_shapes[ndim]:
                for crop_shape in crop_shapes[ndim]:
                    for use_labels in [True, False]:
                        yield check_random_bbox_crop_overlap, \
                                batch_size, ndim, crop_shape, input_shape, use_labels

def test_random_bbox_crop_no_labels():
    batch_size = 3
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    test_box_shape = [200, 4]
    def get_boxes():
        out = [(np.random.randint(0, 255, size = test_box_shape, dtype = np.uint8) / 255).astype(dtype = np.float32) for _ in range(batch_size)]
        return out
    boxes = fn.external_source(source = get_boxes)
    processed = fn.random_bbox_crop(boxes,
                                    aspect_ratio=[0.5, 2.0],
                                    thresholds=[0.1, 0.3, 0.5],
                                    scaling=[0.8, 1.0],
                                    bbox_layout="xyXY")
    pipe.set_outputs(*processed)
    pipe.build()
    for _ in range(3):
        pipe.run()
