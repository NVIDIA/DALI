import nvidia.dali as dali


def input(
    tfrecord_files, tfrecord_idxs, device, shard_id, num_shards, random_shuffle=True
):
    inputs = dali.fn.readers.tfrecord(
        path=tfrecord_files,
        index_path=tfrecord_idxs,
        features={
            "image/encoded": dali.tfrecord.FixedLenFeature(
                (), dali.tfrecord.string, ""
            ),
            "image/source_id": dali.tfrecord.FixedLenFeature(
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
            "image/object/area": dali.tfrecord.VarLenFeature(
                dali.tfrecord.float32, 0.0
            ),
            #'image/object/is_crowd': dali.tfrecord.VarLenFeature(dali.tfrecord.int64, 0)
        },
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
    )

    images = dali.fn.decoders.image(
        inputs["image/encoded"],
        device="cpu",
        # device='mixed' if device == 'gpu' else 'cpu',
        output_type=dali.types.RGB,
    )
    xmin = inputs["image/object/bbox/xmin"]
    xmax = inputs["image/object/bbox/xmax"]
    ymin = inputs["image/object/bbox/ymin"]
    ymax = inputs["image/object/bbox/ymax"]
    bboxes = dali.fn.transpose(
        dali.fn.stack(xmin, ymin, xmax, ymax), perm=[1, 0], device="cpu"
    )
    classes = dali.fn.cast(
        inputs["image/object/class/label"], dtype=dali.types.INT32, device="cpu"
    )
    return images, bboxes, classes


def normalize_flip(device, images, bboxes, p=0.5):
    flip = dali.fn.random.coin_flip(probability=p)
    images = dali.fn.crop_mirror_normalize(
        images,
        mirror=flip,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout=dali.types.NHWC,
        device="cpu",  # device
    )
    bboxes = dali.fn.bb_flip(bboxes, horizontal=flip, ltrb=True, device="cpu")
    return images, bboxes


def random_crop_resize_2(
    device, images, bboxes, classes, output_size, scaling=[0.1, 2.0]
):

    scale_factor = dali.fn.random.uniform(range=scaling)
    scaled_x = scale_factor * output_size[0]
    scaled_y = scale_factor * output_size[1]

    input_size = dali.fn.shapes(images, dtype=dali.types.INT32, device="cpu")
    width = dali.fn.slice(input_size, 1, 1, axes=[0], device="cpu")
    height = dali.fn.slice(input_size, 0, 1, axes=[0], device="cpu")
    image_scale = dali.math.min(scaled_x / width, scaled_y / height)

    scaled_width = width * image_scale
    scaled_height = height * image_scale

    images = dali.fn.resize(
        images, resize_x=scaled_width, resize_y=scaled_height, device="cpu"
    )

    crop_shape = dali.fn.constant(idata=output_size, device="cpu")

    anchors, shapes, bboxes, classes = dali.fn.random_bbox_crop(
        bboxes,
        classes,
        crop_shape=crop_shape,
        input_shape=dali.fn.cast(
            dali.fn.cat(scaled_width, scaled_height, device="cpu"),
            dtype=dali.types.INT32,
            device="cpu",
        ),
        bbox_layout="xyXY",
        allow_no_crop=False,
    )
    anchors = dali.fn.cast(anchors, dtype=dali.types.INT32)
    shapes = dali.fn.cast(shapes, dtype=dali.types.INT32)
    images = dali.fn.slice(images, anchors, shapes, out_of_bounds_policy="pad")

    return images, bboxes, classes
