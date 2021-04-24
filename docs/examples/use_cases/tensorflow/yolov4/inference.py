import numpy as np
import tensorflow as tf

import utils


def infer(model, cls_names, input):

    pred_boxes = [[] for i in range(len(cls_names))]
    output = [utils.decode_layer(layer, i) for i, layer in enumerate(model.predict(input))]

    for preds in output:
        xywh, obj, conf = preds
        ltrb = utils.xywh_to_ltrb(xywh)

        objectness = tf.math.reduce_max(conf, axis=-1) * obj
        clss = tf.argmax(conf, axis=-1)

        detected = tf.where(objectness > 0.25)
        for idx in detected:
            batch, ix, iy, ir = idx
            score = objectness[batch, ix, iy, ir].numpy()
            cls = clss[batch, ix, iy, ir]
            box = ltrb[batch, ix, iy, ir]
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
    for cls in range(len(cls_names)):
        cls_preds = sorted(pred_boxes[cls])
        while len(cls_preds) > 0:
            score, box = cls_preds[-1]
            box_xywh = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
            boxes.append(box_xywh)
            scores.append(score)
            labels.append(cls_names[cls])
            rem = []
            for score2, box2 in cls_preds:
                if iou(box, box2) < 0.213:
                    rem.append((score2, box2))
            cls_preds = rem

    return boxes, scores, labels
