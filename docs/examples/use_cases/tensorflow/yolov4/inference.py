from utils import decode_layer
import numpy as np


def infer(model, cls_names, input):

    # TODO: implement as a tf function
    pred_boxes = [[] for i in range(len(cls_names))]
    output = [decode_layer(layer, i) for i, layer in enumerate(model.predict(input))]
    for i, preds in enumerate(output):
        xywh, obj, conf = preds
        gw, gh = xywh.shape[1:3]
        for ix in range(gw):
            for iy in range(gh):
                for ir in range(3):
                    x, y, w, h = xywh[0, ix, iy, ir]

                    cls = np.argmax(conf[0, ix, iy, ir])
                    objectness = conf[0, ix, iy, ir, cls] * obj[0, ix, iy, ir]

                    if objectness > 0.25:
                        l, t, r, b = x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h
                        pred_boxes[cls].append((objectness.numpy(), [l, t, r, b]))

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
