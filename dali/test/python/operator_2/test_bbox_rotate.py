# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.types import DALIDataType
from PIL import Image, ImageDraw
from test_utils import get_dali_extra_path

test_data_root = Path(get_dali_extra_path())
file_root = test_data_root / "db" / "coco" / "images"
train_annotations = test_data_root / "db" / "coco" / "instances.json"

_PIPE_ARGS = {"num_threads": 1, "batch_size": 1, "device_id": 0}


def show_outputs(images: np.ndarray, boxes: np.ndarray, labels: np.ndarray, filename: str):
    image = Image.fromarray(images)
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(label), fill="red")

    image.save(filename)


@pipeline_def(**_PIPE_ARGS)
def coco_rotate_pipeline_visual(angle=45.0, keep_size=False, ltrb=True, ratio=False):
    image_enc, boxes, labels = fn.readers.coco(
        file_root=str(file_root), annotations_file=str(train_annotations), ltrb=ltrb, ratio=ratio
    )
    image = fn.decoders.image(image_enc, device="cpu")
    image_rot = fn.rotate(image, angle=angle, keep_size=keep_size)
    boxes_rot, labels_rot = fn.bbox_rotate(
        boxes,
        labels,
        angle=angle,
        input_shape=image.shape(),
        bbox_layout="xyXY" if ltrb else "xyWH",
        keep_size=keep_size,
        bbox_normalized=ratio,
    )
    return image, boxes, labels, image_rot, boxes_rot, labels_rot


def bbox_rotate_visual():
    """Prefix with 'test' if you want to run this ad-hoc and write image results"""
    angle = 45.0
    ratio = True
    ltrb = True
    keep_size = True
    pipeline = coco_rotate_pipeline_visual(angle=angle, ratio=ratio, ltrb=ltrb, keep_size=keep_size)
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
def coco_rotate_pipeline(angle=45.0, keep_size=False, ltrb=True, ratio=False):
    image_enc, boxes, labels = fn.readers.coco(
        file_root=str(file_root), annotations_file=str(train_annotations), ltrb=ltrb, ratio=ratio
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
        im_ = fn.rotate(im_, angle=angle, keep_size=keep_size, fill_value=0)
        box_ = fn.bbox_rotate(
            box_,
            angle=angle,
            input_shape=im_.shape(),
            keep_size=keep_size,
            remove_threshold=threshold,
            bbox_layout="xyXY",
            bbox_normalized=False,
        )
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
