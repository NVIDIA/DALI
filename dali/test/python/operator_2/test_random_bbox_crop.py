# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import itertools
import os
import random
from tempfile import TemporaryFile

import numpy as np
from nvidia.dali import fn, ops, pipeline_def, types
from nvidia.dali.pipeline import Pipeline

bbox_2d_ltrb_1 = [0.0123, 0.0123, 0.2123, 0.2123]
bbox_2d_ltrb_2 = [0.1123, 0.1123, 0.19123, 0.19123]
bbox_2d_ltrb_3 = [0.3123, 0.3123, 0.5123, 0.5123]
bbox_3d_ltrb_1 = [0.123, 0.6123, 0.6123, 0.7123, 0.7123, 0.7123]
bbox_3d_ltrb_2 = [0.1123, 0.1123, 0.1123, 0.2123, 0.2123, 0.2123]
bbox_3d_ltrb_3 = [0.7123, 0.7123, 0.7123, 0.8123, 0.8123, 0.8123]

bboxes_data = {
    2: [bbox_2d_ltrb_1, bbox_2d_ltrb_2, bbox_2d_ltrb_3],
    3: [bbox_3d_ltrb_1, bbox_3d_ltrb_2, bbox_3d_ltrb_3],
}


class RedirectStdErr:
    """Redirect stderr to a temporary file and return its handle.
    contextlib.redirect_stderr doesn't work for std::cerr logs from DALI."""

    def __init__(self):
        self._f = TemporaryFile(mode="w+")
        self._target_fd = self._f.fileno()
        self._original_stderr_fd = None

    def __enter__(self):
        assert not self._f.closed, "Temporary file is already closed"
        self._original_stderr_fd = os.dup(2)
        os.dup2(self._target_fd, 2)
        return self

    def readlines(self):
        """Get the lines written to stderr since the beginning of this context manager"""
        assert not self._f.closed, "Temporary file is already closed, probably after __exit__"
        self._f.seek(0)
        return self._f.read().splitlines()

    def __exit__(self, *_args):
        try:
            if self._original_stderr_fd is not None:
                os.dup2(self._original_stderr_fd, 2)
                os.close(self._original_stderr_fd)
        finally:
            self._f.close()


class BBoxDataIterator:
    def __init__(self, n, batch_size, ndim=2, produce_labels=False):
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
                for _ in range(self.batch_size - 2):
                    boxes.append(np.array([bboxes[1]], dtype=np.float32))
                    labels.append(np.array([2], dtype=np.int32))

        if self.i < self.n:
            self.i = self.i + 1
            outputs = [boxes]
            if self.produce_labels:
                outputs.append(labels)
            return outputs

        self.i = 0
        raise StopIteration

    next = __next__


class RandomBBoxCropSynthDataPipeline(Pipeline):
    def __init__(
        self,
        *,
        device,
        batch_size,
        bbox_source,
        thresholds,
        scaling,
        aspect_ratio,
        threshold_type="iou",
        bbox_layout="xyXY",
        num_attempts=100,
        allow_no_crop=False,
        input_shape=None,
        crop_shape=None,
        bbox_prune_threshold=None,
        all_boxes_above_threshold=False,
        output_bbox_indices=False,
        num_threads=1,
        device_id=0,
        num_gpus=1,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=1234)

        self.device = device
        self.bbox_source = bbox_source

        self.bbox_crop = ops.RandomBBoxCrop(
            device=self.device,
            aspect_ratio=aspect_ratio,
            scaling=scaling,
            thresholds=thresholds,
            threshold_type=threshold_type,
            bbox_layout=bbox_layout,
            num_attempts=num_attempts,
            allow_no_crop=allow_no_crop,
            input_shape=input_shape,
            crop_shape=crop_shape,
            all_boxes_above_threshold=all_boxes_above_threshold,
            output_bbox_indices=output_bbox_indices,
            bbox_prune_threshold=bbox_prune_threshold,
        )

    def define_graph(self):
        inputs = fn.external_source(
            source=self.bbox_source, num_outputs=self.bbox_source.num_outputs
        )
        outputs = self.bbox_crop(*inputs)
        return [inputs[0], *outputs]


def crop_contains(crop_anchor, crop_shape, point):
    ndim = len(crop_shape)
    assert len(crop_shape) == ndim
    assert len(point) == ndim

    point = np.array(point)
    crop_anchor = np.array(crop_anchor)
    crop_shape = np.array(crop_shape)

    if np.any(np.less(point, crop_anchor)) or np.any(np.greater(point, (crop_anchor + crop_shape))):
        return False
    return True


def filter_by_centroid(crop_anchor, crop_shape, bboxes):
    ndim = len(crop_shape)
    indexes = []
    for i, bbox in enumerate(bboxes):
        centroid = [0.5 * (bbox[d] + bbox[ndim + d]) for d in range(ndim)]
        if crop_contains(crop_anchor, crop_shape, centroid):
            indexes.append(i)
    return np.array(bboxes[indexes, :])


def intersection(bbox_a: np.ndarray, bbox_b: np.ndarray):
    """bbox_a and bbox_b are xy(z)XY(Z) format"""
    ndim, rem = divmod(len(bbox_a), 2)
    assert rem == 0, "Should be even dims"
    xy = np.maximum(bbox_a[:ndim], bbox_b[:ndim])
    XY = np.minimum(bbox_a[ndim:], bbox_b[ndim:])
    sides = XY - xy
    return 0 if any(sides < 0) else np.prod(sides)


def test_intersection():
    a = np.array([0, 0, 2, 2])
    b = np.array([2, 2, 4, 4])
    h = np.array([3, 3, 4, 4])
    assert intersection(a, h) == 0
    assert intersection(a, b) == 0
    assert intersection(b, a) == 0
    assert intersection(b, b) == 4
    c = np.array([1, 1, 4, 4])
    assert intersection(a, c) == 1
    assert intersection(b, c) == 4
    d = np.array([0, 0, 2, 3])
    e = np.array([1, 1, 3, 2])
    assert intersection(d, e) == 1
    assert intersection(e, d) == 1
    f = np.array([0, 0, 0, 3, 3, 3])
    g = np.array([1, 1, 1, 4, 4, 4])
    assert intersection(f, g) == 8


def filter_by_area(crop_anchor, crop_shape, bboxes, thresh):
    ndim = len(crop_shape)
    crop_box = np.concatenate([crop_anchor, crop_shape])  # xywh
    crop_box[ndim:] += crop_box[:ndim]  # xyXY
    indexes = []
    for i, bbox in enumerate(bboxes):
        intersec = intersection(bbox, crop_box)
        box_area = np.prod(bbox[ndim:] - bbox[:ndim])
        if intersec != 0 and intersec / box_area >= thresh:
            indexes.append(i)
    return np.array(bboxes[indexes, :])


def map_box(bbox, crop_anchor, crop_shape):
    ndim = int(len(bbox) / 2)
    assert len(crop_anchor) == ndim
    assert len(crop_shape) == ndim
    new_bbox = np.array(bbox)
    for d in range(ndim):
        c_start = crop_anchor[d]
        c_end = crop_anchor[d] + crop_shape[d]
        b_start = bbox[d]
        b_end = bbox[ndim + d]
        rel_extent = c_end - c_start
        n_start = (max(c_start, b_start) - c_start) / rel_extent
        n_end = (min(c_end, b_end) - c_start) / rel_extent
        new_bbox[d] = max(0.0, min(1.0, n_start))
        new_bbox[ndim + d] = max(0.0, min(1.0, n_end))
    return new_bbox


def check_processed_bboxes(
    crop_anchor, crop_shape, original_boxes, processed_boxes, filter_fn, bbox_indices=None
):
    if bbox_indices is not None:
        filtered_boxes = np.array(original_boxes[bbox_indices])
    else:
        filtered_boxes = filter_fn(crop_anchor, crop_shape, original_boxes)
    assert len(original_boxes) >= len(filtered_boxes)
    assert len(filtered_boxes) == len(processed_boxes)
    nboxes = len(filtered_boxes)
    for i in range(nboxes):
        box = filtered_boxes[i]
        processed_box = processed_boxes[i]
        expected_box = map_box(box, crop_anchor, crop_shape)
        assert np.allclose(expected_box, processed_box, atol=1e-6)


def check_crop_dims_variable_size(anchor, shape, scaling, aspect_ratio):
    ndim = len(shape)
    k = 0
    nranges = len(aspect_ratio) / 2

    max_extent = 0.0
    for d in range(ndim):
        max_extent = shape[d] if shape[d] > max_extent else max_extent
    assert max_extent >= scaling[0] or np.isclose(max_extent, scaling[0])
    assert max_extent <= scaling[1] or np.isclose(max_extent, scaling[1])
    for d in range(ndim):
        assert anchor[d] >= 0.0 and anchor[d] <= 1.0, anchor
        assert anchor[d] + shape[d] > 0.0 and anchor[d] + shape[d] <= 1.0

        for d2 in range(d + 1, ndim):
            ar = shape[d] / shape[d2]
            ar_min = aspect_ratio[k * 2]
            ar_max = aspect_ratio[k * 2 + 1]
            if ar_min == ar_max:
                assert np.isclose(ar, ar_min), f"{ar=}={d}/{d2} is not close to ar_min={ar_min=}"
            else:
                assert aspect_ratio[k * 2] <= ar <= aspect_ratio[k * 2 + 1]
            k = int((k + 1) % nranges)


def check_crop_dims_fixed_size(anchor, shape, expected_crop_shape, input_shape):
    ndim = len(shape)
    for d in range(ndim):
        anchor_rng = sorted((0.0, input_shape[d] - expected_crop_shape[d]))
        assert (
            anchor[d] >= anchor_rng[0] and anchor[d] <= anchor_rng[1]
        ), f"Expected anchor[{d}] to be within the range {anchor_rng}. Got: {anchor[d]}"
        assert shape[d] == expected_crop_shape[d], "{} != {}".format(shape, expected_crop_shape)


def check_random_bbox_crop_variable_shape(
    batch_size,
    ndim,
    scaling,
    aspect_ratio,
    use_labels,
    output_bbox_indices,
    bbox_prune_threshold,
):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(
        device="cpu",
        thresholds=[0, 0.01, 0.05, 0.1, 0.15],
        batch_size=batch_size,
        bbox_source=bbox_source,
        bbox_layout=bbox_layout,
        scaling=scaling,
        aspect_ratio=aspect_ratio,
        input_shape=None,
        crop_shape=None,
        output_bbox_indices=output_bbox_indices,
        bbox_prune_threshold=bbox_prune_threshold,
    )

    if bbox_prune_threshold is None:
        filter_fn = filter_by_centroid
    else:
        filter_fn = functools.partial(filter_by_area, thresh=bbox_prune_threshold)

    for _ in range(100):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_boxes = outputs[0].at(sample)
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            out_boxes = outputs[3].at(sample)
            check_crop_dims_variable_size(out_crop_anchor, out_crop_shape, scaling, aspect_ratio)
            bbox_indices_out_idx = 4 if not use_labels else 5
            bbox_indices = outputs[bbox_indices_out_idx].at(sample) if output_bbox_indices else None
            check_processed_bboxes(
                out_crop_anchor, out_crop_shape, in_boxes, out_boxes, filter_fn, bbox_indices
            )


def test_random_bbox_crop_variable_shape():
    random.seed(1234)
    aspect_ratio_ranges = {
        2: [[0.01, 100], [0.5, 2.0], [1.0, 1.0]],
        3: [[0.5, 2.0, 0.6, 2.1, 0.4, 1.9], [1.0, 1.0], [0.5, 0.5, 0.25, 0.25, 0.5, 0.5]],
    }
    scalings = [[0.3, 0.5], [0.1, 0.3], [0.9, 0.99]]
    for batch_size, ndim, scaling, prune_thresh in itertools.product(
        [3], [2, 3], scalings, [None, 0.0, 0.1, 0.3, 0.5]
    ):
        for aspect_ratio in aspect_ratio_ranges[ndim]:
            use_labels = random.choice([True, False])
            out_bbox_indices = random.choice([True, False])
            yield (
                check_random_bbox_crop_variable_shape,
                batch_size,
                ndim,
                scaling,
                aspect_ratio,
                use_labels,
                out_bbox_indices,
                prune_thresh,
            )


def check_random_bbox_crop_fixed_shape(
    batch_size, ndim, crop_shape, input_shape, use_labels, bbox_prune_threshold
):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(
        device="cpu",
        thresholds=[0, 0.01, 0.05, 0.1, 0.15],
        batch_size=batch_size,
        bbox_source=bbox_source,
        bbox_layout=bbox_layout,
        scaling=None,
        aspect_ratio=None,
        input_shape=input_shape,
        crop_shape=crop_shape,
        all_boxes_above_threshold=False,
        bbox_prune_threshold=bbox_prune_threshold,
    )

    if bbox_prune_threshold is None:
        filter_fn = filter_by_centroid
    else:
        filter_fn = functools.partial(filter_by_area, thresh=bbox_prune_threshold)

    for _ in range(100):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_boxes = outputs[0].at(sample)
            out_crop_anchor = outputs[1].at(sample)
            out_crop_shape = outputs[2].at(sample)
            out_boxes = outputs[3].at(sample)
            check_crop_dims_fixed_size(out_crop_anchor, out_crop_shape, crop_shape, input_shape)
            rel_out_crop_anchor = [out_crop_anchor[d] / input_shape[d] for d in range(ndim)]
            rel_out_crop_shape = [out_crop_shape[d] / input_shape[d] for d in range(ndim)]
            check_processed_bboxes(
                rel_out_crop_anchor, rel_out_crop_shape, in_boxes, out_boxes, filter_fn
            )


def test_random_bbox_crop_fixed_shape():
    input_shapes = {2: [[400, 300]], 3: [[400, 300, 64]]}

    crop_shapes = {
        2: [[100, 50], [400, 300], [600, 400]],
        3: [[100, 50, 32], [400, 300, 64], [600, 400, 48]],
    }
    for batch_size, ndim, prune_thresh in itertools.product(
        [3], [2, 3], [None, 0.0, 0.1, 0.3, 0.5]
    ):
        for input_shape, crop_shape, use_labels in itertools.product(
            input_shapes[ndim], crop_shapes[ndim], [True, False]
        ):
            yield (
                check_random_bbox_crop_fixed_shape,
                batch_size,
                ndim,
                crop_shape,
                input_shape,
                use_labels,
                prune_thresh,
            )


def check_random_bbox_crop_overlap(batch_size, ndim, crop_shape, input_shape, use_labels):
    bbox_source = BBoxDataIterator(100, batch_size, ndim, produce_labels=use_labels)
    bbox_layout = "xyzXYZ" if ndim == 3 else "xyXY"
    pipe = RandomBBoxCropSynthDataPipeline(
        device="cpu",
        batch_size=batch_size,
        thresholds=[1.0],
        threshold_type="overlap",
        num_attempts=1000,
        bbox_source=bbox_source,
        bbox_layout=bbox_layout,
        scaling=None,
        aspect_ratio=None,
        input_shape=input_shape,
        crop_shape=crop_shape,
        all_boxes_above_threshold=False,
    )
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
                    if (
                        rel_out_crop_anchor[d] > box[d]
                        or (rel_out_crop_anchor[d] + rel_out_crop_shape[d]) < box[ndim + d]
                    ):
                        is_box_in = False
                        break

                if is_box_in:
                    at_least_one_box_in = True
                    break
            assert at_least_one_box_in


def test_random_bbox_crop_overlap():
    input_shapes = {2: [[400, 300]], 3: [[400, 300, 64]]}
    crop_shapes = {2: [[150, 150], [400, 300]], 3: [[50, 50, 32], [400, 300, 64]]}
    for batch_size, ndim in itertools.product([3], [2, 3]):
        for input_shape, crop_shape, use_labels in itertools.product(
            input_shapes[ndim], crop_shapes[ndim], [True, False]
        ):
            yield (
                check_random_bbox_crop_overlap,
                batch_size,
                ndim,
                crop_shape,
                input_shape,
                use_labels,
            )


def test_random_bbox_crop_no_labels():
    batch_size = 3
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    test_box_shape = [200, 4]

    def get_boxes():
        out = [
            (np.random.randint(0, 255, size=test_box_shape, dtype=np.uint8) / 255).astype(
                dtype=np.float32
            )
            for _ in range(batch_size)
        ]
        return out

    boxes = fn.external_source(source=get_boxes)
    processed = fn.random_bbox_crop(
        boxes,
        aspect_ratio=[0.5, 2.0],
        thresholds=[0.1, 0.3, 0.5],
        scaling=[0.8, 1.0],
        bbox_layout="xyXY",
    )
    pipe.set_outputs(*processed)
    for _ in range(3):
        pipe.run()


def test_crop_window_warning():
    n_boxes = [10, 12, 0]
    pipe = Pipeline(batch_size=len(n_boxes), num_threads=1, device_id=0, prefetch_queue_depth=1)

    def get_boxes():
        return [np.random.uniform(0, 1, size=(n, 4)).astype(np.float32) for n in n_boxes]

    boxes = fn.external_source(source=get_boxes)
    n_attempt = 10
    processed = fn.random_bbox_crop(
        boxes,
        thresholds=[1.0],  # Impossible threshold to force the warning
        bbox_layout="xyXY",
        allow_no_crop=False,
        total_num_attempts=n_attempt,
    )
    pipe.set_outputs(*processed)
    with RedirectStdErr() as stderr:
        pipe.run()
        logs = stderr.readlines()
    n_lines_expected = sum(n > 0 for n in n_boxes)
    assert (
        len(logs) == n_lines_expected
    ), f"Expected {n_lines_expected} lines of output (one for each non-empty sample)"

    expect_str = (
        "Could not find a valid cropping window to satisfy the specified requirements (attempted"
        f" {n_attempt} times). Using the best cropping window so far (best_metric=0)"
    )
    assert all(
        line.endswith(expect_str) for line in logs
    ), f"Not all lines match expected: {expect_str}"


def test_empty_sample_shape():
    pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)

    def get_boxes():
        return [np.random.uniform(0, 1, size=(0, 4)).astype(np.float32)]

    boxes = fn.external_source(source=get_boxes)
    processed = fn.random_bbox_crop(
        boxes,
        thresholds=[1.0],
        bbox_layout="xyXY",
        allow_no_crop=False,
        input_shape=[600, 400],
        crop_shape=[200, 200],
    )
    pipe.set_outputs(*processed)
    with RedirectStdErr() as stderr:
        for _ in range(3):
            anchor, shape, boxes = pipe.run()
            assert np.all(boxes.shape() == [(0, 4)])
            assert np.all(shape.as_array() == [200, 200])
            assert np.all(anchor.as_array() <= [400, 200])
        logs = stderr.readlines()

    assert len(logs) == 0, f"Expected no logs for empty samples, but got: {logs}"


def _testimpl_random_bbox_crop_square(use_input_shape):
    batch_size = 3
    bbox_source = BBoxDataIterator(100, batch_size, 2, produce_labels=False)

    @pipeline_def(num_threads=1, batch_size=batch_size, device_id=0, seed=1234)
    def random_bbox_crop_fixed_aspect_ratio():
        in_sh = fn.random.uniform(range=(400, 600), shape=(2,), dtype=types.INT32)
        inputs = fn.external_source(source=bbox_source, num_outputs=bbox_source.num_outputs)
        outputs = fn.random_bbox_crop(
            *inputs,
            device="cpu",
            aspect_ratio=(1.0, 1.0),
            scaling=(0.5, 0.8),
            thresholds=[0.0],
            threshold_type="iou",
            bbox_layout="xyXY",
            total_num_attempts=100,
            allow_no_crop=False,
            input_shape=in_sh if use_input_shape else None,
        )
        return in_sh, outputs[1]

    pipe = random_bbox_crop_fixed_aspect_ratio()
    for _ in range(3):
        outputs = pipe.run()
        for sample in range(batch_size):
            in_shape = outputs[0].at(sample)
            out_crop_shape = outputs[1].at(sample)
            if use_input_shape:
                np.testing.assert_allclose(
                    in_shape[0] * out_crop_shape[0], in_shape[1] * out_crop_shape[1], rtol=1e-06
                )
            else:
                np.testing.assert_allclose(out_crop_shape[0], out_crop_shape[1], rtol=1e-06)


def test_random_bbox_crop_square():
    for use_input_shape in [False, True]:
        yield _testimpl_random_bbox_crop_square, use_input_shape
