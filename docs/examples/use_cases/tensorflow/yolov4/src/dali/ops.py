# Copyright 2021-2023 Kacper Kluk, Piotr Kowalewski. All Rights Reserved.
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

import nvidia.dali as dali

def input(file_root, annotations_file, shard_id, num_threads, device):
    inputs, bboxes, classes = dali.fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
        ltrb=True,
        shard_id=shard_id,
        num_shards=num_threads,
        ratio=True,
        random_shuffle=True
    )
    images = dali.fn.decoders.image(inputs, device=device, output_type=dali.types.RGB)

    return images, bboxes, classes

# Converts ltrb bbox coordinates to xywh, where xy denotes coordinates of a bbox center.
def ltrb_to_xywh(bboxes):
    return dali.fn.coord_transform(
        bboxes,
        M=[0.5,  0.0,  0.5,  0.0,
           0.0,  0.5,  0.0,  0.5,
          -1.0,  0.0,  1.0,  0.0,
           0.0, -1.0,  0.0,  1.0]
    )

# Transforms bbox ltrb coordinates, so that they fit within a window
#   anchored at (pos_x, pos_y) and with a shape (shape_x, shape_y).
def bbox_adjust_ltrb(bboxes, shape_x, shape_y, pos_x, pos_y):
    sx, sy, ex, ey = pos_x, pos_y, shape_x + pos_x, shape_y + pos_y
    MT = dali.fn.transforms.crop(
        to_start=dali.fn.stack(sx, sy, sx, sy),
        to_end=dali.fn.stack(ex, ey, ex, ey)
    )
    return dali.fn.coord_transform(bboxes, MT=MT)

# Selects if_true or if_false tensor based on the predicate value.
# predicate should take integer value equal to either 0 or 1.
# if_true and if_false should be 2D tensors with equal size along axis 1.
# Note: this function is a workaround and should be replaced
# with the dedicated operator once available
def select(predicate, if_true, if_false):
    true_shape = if_true.shape(dtype=dali.types.DALIDataType.INT32)
    false_shape = if_false.shape(dtype=dali.types.DALIDataType.INT32)

    joined = dali.fn.cat(if_true, if_false)
    sh = predicate * true_shape + (1 - predicate) * false_shape

    st = dali.fn.stack(true_shape[0] * (1 - predicate), 0)

    return dali.fn.slice(joined, start=st, shape=sh, axes=[0,1], out_of_bounds_policy="trim_to_shape")

# Based on https://github.com/AlexeyAB/darknet/blob/005513a9db14878579adfbb61083962c99bb0a89/src/image.c#L1297
def color_twist(images):
    def random_value():
        value = dali.fn.random.uniform(range=(1, 1.5))
        coin = dali.fn.random.coin_flip()
        return coin * value + (1.0 - coin) * (1.0 / value)

    return dali.fn.color_twist(images,
        hue=dali.fn.random.uniform(range=(-18.0, 18.0)),
        brightness=random_value(),
        contrast=random_value()
    )

def flip(images, bboxes):
    coin = dali.fn.random.coin_flip()
    images = dali.fn.flip(images, horizontal=coin)
    bboxes = dali.fn.bb_flip(bboxes, horizontal=coin, ltrb=True)
    return images, bboxes


# Performs mosaic using MultiPaste operator.
def mosaic(images, bboxes, labels, image_size):
    def generate_tiles(bboxes, labels, shape_x, shape_y):
        idx = dali.fn.batch_permutation()
        permuted_boxes = dali.fn.permute_batch(bboxes, indices=idx)
        permuted_labels = dali.fn.permute_batch(labels, indices=idx)
        shape = dali.fn.stack(shape_y, shape_x)
        in_anchor, _, bbx, lbl = dali.fn.random_bbox_crop(
            permuted_boxes,
            permuted_labels,
            input_shape=image_size,
            crop_shape=shape,
            shape_layout="HW",
            allow_no_crop=False,
            total_num_attempts=64
        )

        # swap coordinates (x, y) -> (y, x)
        in_anchor = in_anchor[::-1]
        in_anchor_c = dali.fn.cast(in_anchor, dtype=dali.types.DALIDataType.INT32)

        return idx, bbx, lbl, in_anchor_c, shape

    prop0_x = dali.fn.random.uniform(range=(0.2, 0.8))
    prop0_y = dali.fn.random.uniform(range=(0.2, 0.8))
    prop1_x = 1.0 - prop0_x
    prop1_y = 1.0 - prop0_y

    pix0_x = dali.fn.cast(prop0_x * image_size[0], dtype=dali.types.DALIDataType.INT32)
    pix0_y = dali.fn.cast(prop0_y * image_size[1], dtype=dali.types.DALIDataType.INT32)
    pix1_x = image_size[0] - pix0_x
    pix1_y = image_size[1] - pix0_y

    perm_UL, bboxes_UL, labels_UL, in_anchor_UL, size_UL = \
        generate_tiles(bboxes, labels, pix0_x, pix0_y)
    perm_UR, bboxes_UR, labels_UR, in_anchor_UR, size_UR = \
        generate_tiles(bboxes, labels, pix1_x, pix0_y)
    perm_LL, bboxes_LL, labels_LL, in_anchor_LL, size_LL = \
        generate_tiles(bboxes, labels, pix0_x, pix1_y)
    perm_LR, bboxes_LR, labels_LR, in_anchor_LR, size_LR = \
        generate_tiles(bboxes, labels, pix1_x, pix1_y)

    idx = dali.fn.stack(perm_UL, perm_UR, perm_LL, perm_LR)
    out_anchors = dali.fn.stack(
        dali.fn.stack(0, 0),
        dali.fn.stack(0, pix0_x),
        dali.fn.stack(pix0_y, 0),
        dali.fn.stack(pix0_y, pix0_x)
    )
    in_anchors = dali.fn.stack(
        in_anchor_UL, in_anchor_UR, in_anchor_LL, in_anchor_LR
    )
    shapes = dali.fn.stack(
        size_UL, size_UR, size_LL, size_LR
    )


    bboxes_UL = bbox_adjust_ltrb(bboxes_UL, prop0_x, prop0_y, 0.0, 0.0)
    bboxes_UR = bbox_adjust_ltrb(bboxes_UR, prop1_x, prop0_y, prop0_x, 0.0)
    bboxes_LL = bbox_adjust_ltrb(bboxes_LL, prop0_x, prop1_y, 0.0, prop0_y)
    bboxes_LR = bbox_adjust_ltrb(bboxes_LR, prop1_x, prop1_y, prop0_x, prop0_y)
    stacked_bboxes = dali.fn.cat(bboxes_UL, bboxes_UR, bboxes_LL, bboxes_LR)
    stacked_labels = dali.fn.cat(labels_UL, labels_UR, labels_LL, labels_LR)

    mosaic = dali.fn.multi_paste(images, in_ids=idx, output_size=image_size, in_anchors=in_anchors,
                                 shapes=shapes, out_anchors=out_anchors, dtype=dali.types.DALIDataType.UINT8)

    return mosaic, stacked_bboxes, stacked_labels
