# Copyright 2021 Kacper Kluk. All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf
import numpy as np
from inference import decode_prediction


ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
SCALES = [1.2, 1.1, 1.05]
SIZE = 608


def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))

def xywh_to_ltrb(boxes):
    boxes = tf.convert_to_tensor(boxes)
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    return tf.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

def ltrb_to_xywh(boxes):
    boxes = tf.convert_to_tensor(boxes)
    l = boxes[..., 0]
    t = boxes[..., 1]
    r = boxes[..., 2]
    b = boxes[..., 3]
    return tf.stack([(l + r) / 2, (t + b) / 2, r - l, b - t], axis=-1)

def calc_ious(boxes1, boxes2):
    ltrb1 = xywh_to_ltrb(boxes1)
    ltrb2 = xywh_to_ltrb(boxes2)

    il = tf.math.maximum(ltrb1[..., 0], ltrb2[..., 0])
    it = tf.math.maximum(ltrb1[..., 1], ltrb2[..., 1])
    ir = tf.math.minimum(ltrb1[..., 2], ltrb2[..., 2])
    ib = tf.math.minimum(ltrb1[..., 3], ltrb2[..., 3])
    I = tf.math.maximum(0.0, ir - il) * tf.math.maximum(0.0, ib - it)

    A1 = (ltrb1[..., 2] - ltrb1[..., 0]) * (ltrb1[..., 3] - ltrb1[..., 1])
    A2 = (ltrb2[..., 2] - ltrb2[..., 0]) * (ltrb2[..., 3] - ltrb2[..., 1])
    U = A1 + A2 - I

    return I / U

def calc_gious(boxes1, boxes2):
    ltrb1 = xywh_to_ltrb(boxes1)
    ltrb2 = xywh_to_ltrb(boxes2)

    il = tf.math.maximum(ltrb1[..., 0], ltrb2[..., 0])
    it = tf.math.maximum(ltrb1[..., 1], ltrb2[..., 1])
    ir = tf.math.minimum(ltrb1[..., 2], ltrb2[..., 2])
    ib = tf.math.minimum(ltrb1[..., 3], ltrb2[..., 3])
    I = tf.math.maximum(0.0, ir - il) * tf.math.maximum(0.0, ib - it)

    A1 = (ltrb1[..., 2] - ltrb1[..., 0]) * (ltrb1[..., 3] - ltrb1[..., 1])
    A2 = (ltrb2[..., 2] - ltrb2[..., 0]) * (ltrb2[..., 3] - ltrb2[..., 1])
    U = A1 + A2 - I

    cl = tf.math.minimum(ltrb1[..., 0], ltrb2[..., 0])
    ct = tf.math.minimum(ltrb1[..., 1], ltrb2[..., 1])
    cr = tf.math.maximum(ltrb1[..., 2], ltrb2[..., 2])
    cb = tf.math.maximum(ltrb1[..., 3], ltrb2[..., 3])
    C = (cr - cl) * (cb - ct)

    return I / U - (C - U) / C

# split model output into xywh, obj and cls tensors
# output tensor shape: [batch, width, height, 3, ...]
def decode_layer(layer, layer_id):
    shape = layer.shape # [batch, width, height, 3 * (5 + classes)]
    d = shape[3]
    gw, gh = shape[1 : 3]
    stride_x = 1 / gw
    stride_y = 1 / gh
    tile_x = tf.cast(tf.tile(tf.expand_dims(tf.range(gw), axis=0), [gw, 1]), tf.float32)
    tile_y = tf.cast(tf.tile(tf.expand_dims(tf.range(gw), axis=1), [1, gh]), tf.float32)
    output_xywh = []
    output_obj = []
    output_cls = []
    for ir in range(3):
        data = layer[..., (d // 3) * ir : (d // 3) * (ir + 1)]
        dx = data[..., 0]
        dy = data[..., 1]
        dw = data[..., 2]
        dh = data[..., 3]
        x = (sigmoid(dx) * SCALES[layer_id] - 0.5 * (SCALES[layer_id] - 1) + tile_x) * stride_x
        y = (sigmoid(dy) * SCALES[layer_id] - 0.5 * (SCALES[layer_id] - 1) + tile_y) * stride_y
        w = tf.math.exp(dw) * ANCHORS[layer_id][ir][0] / SIZE
        h = tf.math.exp(dh) * ANCHORS[layer_id][ir][1] / SIZE
        output_xywh.append(tf.stack([x, y, w, h], axis=-1))

        output_obj.append(sigmoid(data[..., 4]))
        output_cls.append(sigmoid(data[..., 5 : ]))

    return (tf.stack(output_xywh, axis=-2),
            tf.stack(output_obj, axis=-1),
            tf.stack(output_cls, axis=-2))


# probably slow and works only in eager mode
def calc_mAP(predictions, gt_boxes, num_classes):

    def iou(box1, box2):
        l = max(box1[0], box2[0])
        t = max(box1[1], box2[1])
        r = min(box1[2], box2[2])
        b = min(box1[3], box2[3])
        i = max(0, r - l) * max(0, b - t)
        u = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - i
        return i / u

    batch_size = predictions[0].shape[0]
    num_pred_boxes = 0
    num_gt_boxes = 0
    num_true_positives = 0

    stats = []
    for batch_idx in range(batch_size):
        prediction = tuple(p[batch_idx : batch_idx + 1, ...] for p in predictions)
        pred_boxes, scores, pred_classes = decode_prediction(prediction, num_classes)

        boxes = gt_boxes[batch_idx, :, : 4]
        classes = gt_boxes[batch_idx, :, 4]

        gt_used_idx = []
        num_pred_boxes += len(pred_boxes)
        num_gt_boxes += len(classes)

        for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
            found = False
            for gt_idx, (gt_box, gt_class) in enumerate(zip(boxes, classes)):

                if gt_idx in gt_used_idx:
                    continue

                if pred_class != gt_class:
                    continue

                gt_ltrb = (gt_box[0] - gt_box[2] / 2, gt_box[1] - gt_box[3] / 2,
                    gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2)
                if iou(pred_box, gt_ltrb) < 0.5:
                    continue

                found = True
                num_true_positives += 1
                break

            stats.append((scores[pred_idx], found))

    if num_pred_boxes == 0:
        return 0.0

    ap = 0.0
    max_prec = num_true_positives / num_pred_boxes
    for _, found in sorted(stats):
        if found:
            ap += max_prec / num_gt_boxes
            num_true_positives -= 1
        num_pred_boxes -= 1
        if num_pred_boxes == 0:
            break
        max_prec = max(max_prec, num_true_positives / num_pred_boxes)

    return ap
