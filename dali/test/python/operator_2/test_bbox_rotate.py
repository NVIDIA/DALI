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

import numpy as np
from nvidia.dali import fn
from nvidia.dali.pipeline import pipeline_def
from test_utils import get_dali_extra_path

test_data_root = Path(get_dali_extra_path())
file_root = test_data_root / "db" / "coco" / "images"
train_annotations = test_data_root / "db" / "coco" / "instances.json"


def show_outputs(images: np.ndarray, boxes: np.ndarray, labels: np.ndarray, filename: str):
    from PIL import Image, ImageDraw

    image = Image.fromarray(images)
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(label), fill="red")

    image.save(filename)


@pipeline_def(num_threads=1, batch_size=1, device_id=0)
def coco_rotate_pipeline(angle=45.0, keep_size=False, ltrb=True, ratio=False):
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


def test_bbox_rotate():
    ratio = False
    pipeline = coco_rotate_pipeline(ratio=ratio)
    pipeline.build()
    outs = pipeline.run()
    images, boxes, labels, images_rot, boxes_rot, labels_rot = map(lambda x: np.array(x[0]), outs)
    if ratio:
        images_whwh = np.tile(np.array(images.shape[1::-1])[None], 2)
        boxes *= images_whwh
        images_rot_whwh = np.tile(np.array(images_rot.shape[1::-1])[None], 2)
        boxes_rot *= images_rot_whwh
    show_outputs(images, boxes, labels, "original.webp")
    show_outputs(images_rot, boxes_rot, labels_rot, "rotated.webp")
