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

import utils


def decode_prediction(prediction, num_classes):

    pred_boxes = [[] for i in range(num_classes)]

    for i, layer in enumerate(prediction):
        xywh, obj, conf = utils.decode_layer(layer, i)
        ltrb = utils.xywh_to_ltrb(xywh)

        objectness = tf.math.reduce_max(conf, axis=-1) * obj
        clss = tf.argmax(conf, axis=-1)

        detected = tf.where(objectness > 0.25)
        for idx in detected:
            batch, ix, iy, ir = idx
            score = objectness[batch, ix, iy, ir].numpy()
            cls = clss[batch, ix, iy, ir]
            box = list(ltrb[batch, ix, iy, ir].numpy())
            pred_boxes[cls].append((score, box))

    # nms
    def iou(box1, box2):
        l = max(box1[0], box2[0])
        t = max(box1[1], box2[1])
        r = min(box1[2], box2[2])
        b = min(box1[3], box2[3])
        i = max(0, r - l) * max(0, b - t)
        u = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - i
        return i / u

    boxes = []
    scores = []
    labels = []
    for cls in range(num_classes):
        cls_preds = sorted(pred_boxes[cls])
        while len(cls_preds) > 0:
            score, box = cls_preds[-1]
            boxes.append(box)
            scores.append(score)
            labels.append(cls)
            rem = []
            for score2, box2 in cls_preds:
                if iou(box, box2) < 0.213:
                    rem.append((score2, box2))
            cls_preds = rem

    return boxes, scores, labels
