# Copyright 2021 Kacper Kluk, Paweł Anikiel, Jagoda Kamińska. All Rights Reserved.
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
import math
import nvidia.dali as dali


def input_tfrecord(
    tfrecord_files, tfrecord_idxs, device, shard_id, num_shards, random_shuffle=True
):
    inputs = dali.fn.readers.tfrecord(
        path=tfrecord_files,
        index_path=tfrecord_idxs,
        features={
            "image/encoded": dali.tfrecord.FixedLenFeature(
                (), dali.tfrecord.string, ""
            ),
            "image/height": dali.tfrecord.FixedLenFeature((), dali.tfrecord.int64, -1),
            "image/width": dali.tfrecord.FixedLenFeature((), dali.tfrecord.int64, -1),
            "image/object/bbox/xmin": dali.tfrecord.VarLenFeature(
                dali.tfrecord.float32, 0.0
            ),
            "image/object/bbox/xmax": dali.tfrecord.VarLenFeature(
                dali.tfrecord.float32, 0.0
            ),
            "image/object/bbox/ymin": dali.tfrecord.VarLenFeature(
                dali.tfrecord.float32, 0.0
            ),
            "image/object/bbox/ymax": dali.tfrecord.VarLenFeature(
                dali.tfrecord.float32, 0.0
            ),
            "image/object/class/label": dali.tfrecord.VarLenFeature(
                dali.tfrecord.int64, 0
            ),
        },
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
    )

    images = dali.fn.decoders.image(
        inputs["image/encoded"],
        device="mixed" if device == "gpu" else "cpu",
        output_type=dali.types.RGB,
    )
    xmin = inputs["image/object/bbox/xmin"]
    xmax = inputs["image/object/bbox/xmax"]
    ymin = inputs["image/object/bbox/ymin"]
    ymax = inputs["image/object/bbox/ymax"]
    bboxes = dali.fn.transpose(dali.fn.stack(xmin, ymin, xmax, ymax), perm=[1, 0])
    classes = dali.fn.cast(inputs["image/object/class/label"], dtype=dali.types.INT32)
    return (
        images,
        bboxes,
        classes,
        dali.fn.cast(inputs["image/width"], dtype=dali.types.FLOAT),
        dali.fn.cast(inputs["image/height"], dtype=dali.types.FLOAT),
    )


def input_coco(
    images_path, annotations_path, device, shard_id, num_shards, random_shuffle=True
):
    encoded, bboxes, classes = dali.fn.readers.coco(
        file_root=images_path,
        annotations_file=annotations_path,
        ratio=True,
        ltrb=True,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
    )

    images = dali.fn.decoders.image(
        encoded,
        device="mixed" if device == "gpu" else "cpu",
        output_type=dali.types.RGB,
    )

    shape = dali.fn.peek_image_shape(encoded, dtype=dali.types.FLOAT)
    heights = shape[0]
    widths = shape[1]

    return (
        images,
        bboxes,
        classes,
        widths,
        heights,
    )


def normalize_flip(images, bboxes, p=0.5):
    flip = dali.fn.random.coin_flip(probability=p)
    images = dali.fn.crop_mirror_normalize(
        images,
        mirror=flip,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout=dali.types.NHWC
    )
    bboxes = dali.fn.bb_flip(bboxes, horizontal=flip, ltrb=True)
    return images, bboxes


def gridmask(images, widths, heights):

    p = dali.fn.random.coin_flip()
    ratio = 0.4 * p
    angle = dali.fn.random.normal(mean=-1, stddev=1) * 10.0 * (math.pi / 180.0)

    l = dali.math.min(0.5 * heights, 0.3 * widths)
    r = dali.math.max(0.5 * heights, 0.3 * widths)
    tile = dali.fn.cast(
        (dali.fn.random.uniform(range=[0.0, 1.0]) * (r - l) + l),
        dtype=dali.types.INT32,
    )

    gridmask = dali.fn.grid_mask(
        images, ratio=ratio, angle=angle, tile=tile
    )

    return images


def random_crop_resize(
    images, bboxes, classes, widths, heights, output_size, scaling=[0.1, 2.0]
):

    if scaling is None:
        scale_factor = 1.0
    else:
        scale_factor = dali.fn.random.uniform(range=scaling)

    sizes = dali.fn.stack(heights, widths)
    image_scale = dali.math.min(
        scale_factor * output_size[0] / widths,
        scale_factor * output_size[1] / heights,
    )
    scaled_sizes = dali.math.floor(sizes * image_scale + 0.5)

    images = dali.fn.resize(
        images,
        size=scaled_sizes
    )

    anchors, shapes, bboxes, classes = dali.fn.random_bbox_crop(
        bboxes,
        classes,
        crop_shape=output_size,
        input_shape=dali.fn.cast(scaled_sizes, dtype=dali.types.INT32),
        bbox_layout="xyXY",
        allow_no_crop=False,
        total_num_attempts=64,
    )

    images = dali.fn.slice(
        images,
        anchors,
        shapes,
        normalized_anchor=False,
        normalized_shape=False,
        out_of_bounds_policy="pad"
    )

    return (
        images,
        bboxes,
        classes,
    )


def bbox_to_effdet_format(bboxes, image_size):
    w = image_size[0]
    h = image_size[1]
    M = [0.0,   h, 0.0, 0.0,
           w, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0,   h,
         0.0, 0.0,   w, 0.0]

    return dali.fn.coord_transform(bboxes, M=M)
