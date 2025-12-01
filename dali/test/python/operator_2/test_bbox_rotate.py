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

from pathlib import Path

import cv2
import numpy as np
from nose2.tools import cartesian_params, params
from nose_utils import assert_raises
from nvidia.dali import fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.types import DALIDataType
from PIL import Image, ImageDraw
from test_utils import get_dali_extra_path, np_type_to_dali

test_data_root = Path(get_dali_extra_path())
file_root = test_data_root / "db" / "coco" / "images"
train_annotations = test_data_root / "db" / "coco" / "instances.json"

_PIPE_ARGS = {"num_threads": 1, "batch_size": 1}


def show_outputs(images: np.ndarray, boxes: np.ndarray, labels: np.ndarray, filename: str):
    image = Image.fromarray(images)
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(label), fill="red")

    image.save(filename)


@pipeline_def(**_PIPE_ARGS)
def coco_rotate_pipeline_visual(
    angle=45.0, keep_size=False, ltrb=True, ratio=False, mode="expand", shape=None
):
    image_enc, boxes, labels = fn.readers.coco(
        file_root=str(file_root), annotations_file=str(train_annotations), ltrb=ltrb, ratio=ratio
    )
    image = fn.decoders.image(image_enc, device="cpu")
    image_rot = fn.rotate(image, angle=angle, keep_size=keep_size, size=shape)
    boxes_rot, labels_rot = fn.bbox_rotate(
        boxes,
        labels,
        angle=angle,
        input_shape=image.shape(),
        bbox_layout="xyXY" if ltrb else "xyWH",
        keep_size=keep_size,
        bbox_normalized=ratio,
        mode=mode,
        size=shape,
    )
    return image, boxes, labels, image_rot, boxes_rot, labels_rot


def box_rotate_visual():
    """Run this and modify the variables to visualize the output of this operator.

    Example running from repo's root directory:
    `PYTHONPATH=dali/test/python DALI_EXTRA_PATH=/opt/dali_extra/
    python3 -m nose2 -s=dali/test/python/operator_2 test_bbox_rotate.box_rotate_visual`
    """
    angle = 45.0
    ratio = False
    ltrb = False
    shape = (1000, 1000)
    pipeline = coco_rotate_pipeline_visual(
        angle=angle, ratio=ratio, ltrb=ltrb, keep_size=False, mode="expand", shape=shape
    )
    pipeline.build()
    outs = pipeline.run()
    images, boxes, labels, images_rot, boxes_rot, labels_rot = map(lambda x: np.array(x[0]), outs)
    if ratio:
        images_whwh = np.tile(np.array(images.shape[1::-1])[None], 2)
        boxes *= images_whwh
        images_rot_whwh = np.tile(np.array(images_rot.shape[1::-1])[None], 2)
        boxes_rot *= images_rot_whwh

    if not ltrb:
        boxes[..., 2:] += boxes[..., :2]
        boxes_rot[..., 2:] += boxes_rot[..., :2]

    show_outputs(images, boxes, labels, "original.webp")
    show_outputs(images_rot, boxes_rot, labels_rot, "rotated.webp")


@pipeline_def(**_PIPE_ARGS)
def coco_rotate_pipeline(angle=45.0, keep_size=False, ltrb=True, ratio=False, size=None):
    image_enc, boxes, labels = fn.readers.coco(
        file_root=str(file_root),
        annotations_file=str(train_annotations),
        ltrb=ltrb,
        ratio=ratio,
    )
    im_shape = fn.peek_image_shape(image_enc)
    boxes_rot, labels_rot = fn.bbox_rotate(
        boxes,
        labels,
        angle=angle,
        input_shape=im_shape,
        bbox_layout="xyXY" if ltrb else "xyWH",
        keep_size=keep_size,
        bbox_normalized=ratio,
        size=size,
    )
    return im_shape, boxes_rot, labels_rot


@cartesian_params([True, False], [True, False], [True, False])
def test_bbox_rotate(ratio, ltrb, keep_size):
    angle = 45.0
    pipeline = coco_rotate_pipeline(angle=angle, ratio=ratio, ltrb=ltrb, keep_size=keep_size)
    pipeline.build()
    outs = pipeline.run()
    im_hwc, boxes_rot, _ = map(lambda x: np.array(x[0]), outs)

    if keep_size is False:
        rad = np.deg2rad(angle)
        new_w = np.abs(im_hwc[1] * np.cos(rad)) + np.abs(im_hwc[0] * np.sin(rad))
        new_h = np.abs(im_hwc[0] * np.cos(rad)) + np.abs(im_hwc[1] * np.sin(rad))
        im_hwc[0] = np.round(new_h)
        im_hwc[1] = np.round(new_w)

    if ratio:
        images_rot_whwh = np.tile(np.array(im_hwc[1::-1])[None], 2)
        boxes_rot *= images_rot_whwh

    if not ltrb:
        boxes_rot[..., 2:] += boxes_rot[..., :2]

    if keep_size:
        expected = np.array([[359.986, 195.743, 813.241, 648.999]], dtype=np.float32)
    else:
        expected = np.array([[511.971, 507.729, 965.250, 961.007]], dtype=np.float32)

    np.testing.assert_allclose(boxes_rot, expected, rtol=1e-4, atol=1e-2)


@cartesian_params([True, False], [True, False])
def test_bbox_rotate_canvas(ratio, ltrb):
    angle = 45.0
    pipeline = coco_rotate_pipeline(angle=angle, ratio=ratio, ltrb=ltrb, size=(1000, 1000))
    pipeline.build()
    outs = pipeline.run()
    _, boxes_rot, _ = map(lambda x: np.array(x[0]), outs)

    if ratio:
        boxes_rot *= np.array([1000] * 4, dtype=np.float32)

    if not ltrb:
        boxes_rot[..., 2:] += boxes_rot[..., :2]

    expected = np.array([[219.986, 215.743, 673.241, 668.999]], dtype=np.float32)

    np.testing.assert_allclose(boxes_rot, expected, rtol=1e-4, atol=1e-2)


def draw_box(im: np.ndarray, boxes: np.ndarray):
    """Overlay xyXY image-coordinates boxes as masks on image"""
    impil = Image.fromarray(im)
    imdraw = ImageDraw.Draw(impil)
    for box in boxes:
        imdraw.rectangle(box, fill=1)
    return np.array(impil)


@params(0.4, 0.6)
def test_cull_threshold(threshold):
    """Test box is removed below 0.6 threshold and kept above 0.4 threshold"""
    angle = 45.0
    keep_size = True

    im = np.zeros((128, 128), dtype=np.uint8)
    boxes = np.array(((0, 0, 36, 36),), dtype=np.float32)
    im = draw_box(im, boxes)

    def datasource():
        # Add batch and channel dim for dataloader (fn.rotate needs HWC)
        yield im[None, ..., None], boxes[None]

    @pipeline_def(**_PIPE_ARGS)
    def datapipe():
        im_, box_ = fn.external_source(
            datasource,
            num_outputs=2,
            layout=["HWC", ""],
            dtype=[DALIDataType.UINT8, DALIDataType.FLOAT],
        )
        box_ = fn.bbox_rotate(
            box_,
            angle=angle,
            input_shape=im_.shape(),
            keep_size=keep_size,
            remove_threshold=threshold,
            bbox_layout="xyXY",
            bbox_normalized=False,
        )
        im_ = fn.rotate(im_, angle=angle, keep_size=keep_size, fill_value=0)
        return im_, box_

    pipe = datapipe()
    pipe.build()
    outs = pipe.run()
    im_rot, box_out = map(lambda x: np.array(x[0]), outs)

    # 45 deg scales h and w by sqrt2
    old_box_area = np.prod(boxes[..., 2:]) * (np.sqrt(2) ** 2)
    # Get actual box with bounding rect of rotated image
    box_rot = cv2.boundingRect(im_rot)
    clip_box_area = box_rot[-1] * box_rot[-2]
    # Check the threshold change
    box_area_change = clip_box_area / old_box_area
    should_keep = box_area_change > threshold

    if threshold == 0.4:
        assert should_keep
    elif threshold == 0.6:
        assert not should_keep
    else:
        raise RuntimeError("unexpected threshold used")

    if should_keep:
        assert box_out.shape[0] > 0
    else:
        assert box_out.shape[0] == 0


@cartesian_params(("expand", "halfway", "fixed"), (45.0, 90.0))
def test_box_expansion_control(expansion: str, angle: float):
    """Tests the bounding box expansion controls, ensuring aspect ratio flipping is recognised"""
    keep_size = True

    im = np.zeros((128, 128), dtype=np.uint8)
    boxes = np.array(((32, 48, 96, 80),), dtype=np.float32)
    im = draw_box(im, boxes)

    def datasource():
        # Add batch and channel dim for dataloader (fn.rotate needs HWC)
        yield im[None, ..., None], boxes[None]

    @pipeline_def(**_PIPE_ARGS)
    def datapipe():
        im_, box_ = fn.external_source(
            datasource,
            num_outputs=2,
            layout=["HWC", ""],
            dtype=[DALIDataType.UINT8, DALIDataType.FLOAT],
        )
        box_ = fn.bbox_rotate(
            box_,
            angle=angle,
            input_shape=im_.shape(),
            keep_size=keep_size,
            bbox_layout="xyXY",
            bbox_normalized=False,
            mode=expansion,
        )
        return box_

    pipe = datapipe()
    pipe.build()
    (out,) = pipe.run()
    box_out = np.array(out[0])  # Remove batch
    out_box_wh = box_out[..., 2:] - box_out[..., :2]
    in_box_wh = boxes[..., 2:] - boxes[..., :2]

    flipped_aspect = 45 < angle < 135 or 225 < angle < 315

    rad = np.deg2rad(angle)
    expansion_mat = np.abs(np.array([[np.cos(rad), np.sin(rad)], [np.sin(rad), np.cos(rad)]]))
    exp_box_wh = in_box_wh @ expansion_mat
    if expansion == "expand":
        np.testing.assert_allclose(out_box_wh, exp_box_wh)
    elif expansion == "halfway":
        if flipped_aspect:  # Flip wh when aspect ratio changes
            in_box_wh = in_box_wh[..., ::-1]
        diff = exp_box_wh - in_box_wh
        exp_box_wh -= diff / 2
        np.testing.assert_allclose(out_box_wh, exp_box_wh, rtol=1e-4, atol=1e-3)
    elif expansion == "fixed":
        if flipped_aspect:  # Flip wh when aspect ratio changes
            in_box_wh = in_box_wh[..., ::-1]
        np.testing.assert_allclose(out_box_wh, in_box_wh, rtol=1e-4, atol=1e-3)


@params([200], [200, 1], [200, 1, 1], [180, 1])
def test_allowed_label_shapes(test_labels_shape: list[int]):
    """Ensure labels can have N or Nx1 but not anything else"""
    num_boxes = 200

    def get_boxes():
        out = [np.random.randint(0, 255, size=[num_boxes, 4]).astype(np.float32) for _ in range(1)]
        return out

    def get_labels():
        out = [np.random.randint(0, 255, size=test_labels_shape, dtype=np.int32) for _ in range(1)]
        return out

    @pipeline_def(**_PIPE_ARGS)
    def datapipe():
        boxes = fn.external_source(source=get_boxes)
        labels = fn.external_source(source=get_labels)
        boxes, labels = fn.bbox_rotate(
            boxes,
            labels,
            angle=45,
            input_shape=[255, 255],
            bbox_layout="xyXY",
            bbox_normalized=False,
        )
        return boxes, labels

    if len(test_labels_shape) > 2 or test_labels_shape[0] != num_boxes:
        with assert_raises(RuntimeError):
            pipe = datapipe()
            pipe.build()
            pipe.run()
    else:
        pipe = datapipe()
        pipe.build()
        pipe.run()


@params(np.int32, np.int64, np.uint32, np.float32, np.float64, np.uint8)
def test_box_tensor_sizes(dtype):
    """Test that size argument can be passed as a tensor of various types"""

    def get_boxes():
        out = [np.random.randint(0, 255, size=[10, 4]).astype(np.float32) for _ in range(1)]
        return out

    def get_shape():
        out = [np.array([128, 128], dtype=dtype) for _ in range(1)]
        return out

    @pipeline_def(**_PIPE_ARGS)
    def datapipe():
        boxes = fn.external_source(source=get_boxes)
        shape = fn.external_source(source=get_shape, dtype=np_type_to_dali(dtype))
        boxes = fn.bbox_rotate(boxes, angle=45, input_shape=[255, 255], size=shape)
        return boxes

    if dtype in {np.uint8, np.float64}:
        with assert_raises(RuntimeError):
            pipe = datapipe()
            pipe.build()
            pipe.run()
    else:
        pipe = datapipe()
        pipe.build()
        pipe.run()
