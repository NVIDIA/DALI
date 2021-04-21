import numpy as np
import tensorflow as tf
import numpy as np


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
