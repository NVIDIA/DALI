import nvidia.dali as dali

def input(file_root, annotations_file, device_id, num_threads, device, random_shuffle=True):
    inputs, bboxes, classes = dali.fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
        ltrb=True,
        shard_id=device_id,
        num_shards=num_threads,
        ratio=True,
        random_shuffle=random_shuffle
    )
    images = dali.fn.decoders.image(inputs, device=device, output_type=dali.types.RGB)

    return images, bboxes, classes


def permute(images, bboxes, labels):
    indices = dali.fn.batch_permutation()
    return (
        dali.fn.permute_batch(images, indices=indices),
        dali.fn.permute_batch(bboxes, indices=indices),
        dali.fn.permute_batch(labels, indices=indices),
    )

def generate_tiles(images, bboxes, labels, shape_x, shape_y, image_size):
    images, bboxes, labels = permute(images, bboxes, labels)
    crop_anchor, crop_shape, bboxes, labels = dali.fn.random_bbox_crop(
        bboxes,
        labels,
        crop_shape=dali.fn.stack(shape_x, shape_y),
        input_shape=image_size,
        bbox_layout="xyWH",
        thresholds=[0.0]
    )
    images = dali.fn.slice(images, crop_anchor, crop_shape, normalized_anchor=False, normalized_shape=False)
    return images, bboxes, labels

def xywh_to_ltrb(bboxes):
    Z = dali.types.Constant(0.0)
    H = dali.types.Constant(0.5)
    MH = dali.types.Constant(-0.5)
    O = dali.types.Constant(1.0)

    M = dali.fn.stack(
            dali.fn.stack(O, Z, MH, Z),
            dali.fn.stack(Z, O, Z, MH),
            dali.fn.stack(O, Z, H, Z),
            dali.fn.stack(Z, O, Z, H)
        )
    return dali.fn.coord_transform(bboxes, M=M)


def ltrb_to_xywh(bboxes):
    Z = dali.types.Constant(0.0)
    H = dali.types.Constant(0.5)
    O = dali.types.Constant(1.0)
    MO = dali.types.Constant(-1.0)

    M = dali.fn.stack(
            dali.fn.stack(H, Z, H, Z),
            dali.fn.stack(Z, H, Z, H),
            dali.fn.stack(MO, Z, O, Z),
            dali.fn.stack(Z, MO, Z, O)
        )
    return dali.fn.coord_transform(bboxes, M=M)


def bbox_adjust_xywh(bboxes, shape_x, shape_y, pos_x, pos_y):
    Z = dali.types.Constant(0.0)

    M = dali.fn.stack(
            dali.fn.stack(shape_x, Z, Z, Z),
            dali.fn.stack(Z, shape_y, Z, Z),
            dali.fn.stack(Z, Z, shape_x, Z),
            dali.fn.stack(Z, Z, Z, shape_y)
        )

    T = dali.fn.stack(pos_x, pos_y, Z, Z)
    return dali.fn.coord_transform(bboxes, M=M, T=T)

def bbox_adjust_ltrb(bboxes, shape_x, shape_y, pos_x, pos_y):
    sx, sy, ex, ey = pos_x, pos_y, shape_x + pos_x, shape_y + pos_y
    MT = dali.fn.transforms.crop(
        to_start=dali.fn.stack(sx, sy, sx, sy), to_end=dali.fn.stack(ex, ey, ex, ey)
    )
    return dali.fn.coord_transform(bboxes, MT=MT)


def mosaic(images, bboxes, labels, image_size):
    prob_x = dali.fn.uniform(range=(0.2, 0.8))
    prob_y = dali.fn.uniform(range=(0.2, 0.8))

    pix0_x = dali.fn.cast(prob_x * image_size[0], dtype=dali.types.INT32)
    pix0_y = dali.fn.cast(prob_y * image_size[1], dtype=dali.types.INT32)
    pix1_x = image_size[0] - pix0_x
    pix1_y = image_size[1] - pix0_y

    images00, bboxes00, labels00 = generate_tiles(
        images, bboxes, labels, pix0_x, pix0_y, image_size
    )
    images01, bboxes01, labels01 = generate_tiles(
        images, bboxes, labels, pix0_x, pix1_y, image_size
    )
    images10, bboxes10, labels10 = generate_tiles(
        images, bboxes, labels, pix1_x, pix0_y, image_size
    )
    images11, bboxes11, labels11 = generate_tiles(
        images, bboxes, labels, pix1_x, pix1_y, image_size
    )
    images0 = dali.fn.cat(images00, images01, axis=0)
    images1 = dali.fn.cat(images10, images11, axis=0)
    images = dali.fn.cat(images0, images1, axis=1)

    zeros = dali.types.Constant(0.0)
    #bboxes00 = bbox_adjust_lrtb(bboxes00, prob_x, prob_y, zeros, zeros)
    #bboxes01 = bbox_adjust_lrtb(bboxes01, prob_x, 1.0 - prob_y, zeros, prob_y)
    #bboxes10 = bbox_adjust_lrtb(bboxes10, 1.0 - prob_x, prob_y, prob_x, zeros)
    #bboxes11 = bbox_adjust_lrtb(bboxes11, 1.0 - prob_x, 1.0 - prob_y, prob_x, prob_y)
    #bboxes = lrtb_to_xywh(dali.fn.cat(bboxes00, bboxes01, bboxes10, bboxes11))

    bboxes00 = bbox_adjust_xywh(bboxes00, prob_x, prob_y, zeros, zeros)
    bboxes01 = bbox_adjust_xywh(bboxes01, prob_x, 1.0 - prob_y, zeros, prob_y)
    bboxes10 = bbox_adjust_xywh(bboxes10, 1.0 - prob_x, prob_y, prob_x, zeros)
    bboxes11 = bbox_adjust_xywh(bboxes11, 1.0 - prob_x, 1.0 - prob_y, prob_x, prob_y)
    bboxes = dali.fn.cat(bboxes00, bboxes01, bboxes10, bboxes11)


    labels = dali.fn.cat(labels00, labels01, labels10, labels11)

    return images, bboxes, labels


def mosaic_new(images, bboxes, labels, image_size):
    zeros = dali.fn.constant(idata=0, shape=[])
    zeros_f = dali.fn.constant(fdata=0.0, shape=[])
    cuts_x = dali.fn.random.uniform(zeros, range=(0.2, 0.8))
    cuts_y = dali.fn.random.uniform(zeros, range=(0.2, 0.8))

    prop_x = dali.fn.cast(cuts_x * image_size[0], dtype=dali.types.DALIDataType.INT32)
    prop_y = dali.fn.cast(cuts_y * image_size[1], dtype=dali.types.DALIDataType.INT32)

    def generate_tiles(bboxes, labels, shape_x, shape_y):
        idx = dali.fn.batch_permutation()
        permuted_boxes = dali.fn.permute_batch(bboxes, indices=idx)
        permuted_labels = dali.fn.permute_batch(labels, indices=idx)
        shape = dali.fn.stack(shape_y, shape_x)
        in_anchor, in_shape, bbx, lbl = dali.fn.random_bbox_crop(
            permuted_boxes,
            permuted_labels,
            input_shape=image_size,
            crop_shape=shape,
            bbox_layout="xyXY",
            shape_layout="HW",
            allow_no_crop=False,
            total_num_attempts=64
        )

        in_anchor = dali.fn.stack(dali.fn.reductions.sum(in_anchor), dali.fn.reductions.sum(in_anchor)) - in_anchor
        in_anchor_c = dali.fn.cast(in_anchor, dtype=dali.types.DALIDataType.INT32)

        return idx, bbx, lbl, in_anchor_c, shape

    perm_UL, bboxes_UL, labels_UL, in_anchor_UL, size_UL = \
        generate_tiles(bboxes, labels, prop_x, prop_y)
    perm_UR, bboxes_UR, labels_UR, in_anchor_UR, size_UR = \
        generate_tiles(bboxes, labels, image_size[1] - prop_x, prop_y)
    perm_LL, bboxes_LL, labels_LL, in_anchor_LL, size_LL = \
        generate_tiles(bboxes, labels, prop_x, image_size[0] - prop_y)
    perm_LR, bboxes_LR, labels_LR, in_anchor_LR, size_LR = \
        generate_tiles(bboxes, labels, image_size[1] - prop_x, image_size[0] - prop_y)

    idx = dali.fn.stack(perm_UL, perm_UR, perm_LL, perm_LR)
    out_anchors = dali.fn.stack(
        dali.fn.stack(zeros, zeros),
        dali.fn.stack(zeros, prop_x),
        dali.fn.stack(prop_y, zeros),
        dali.fn.stack(prop_y, prop_x)
    )
    in_anchors = dali.fn.stack(
        in_anchor_UL, in_anchor_UR, in_anchor_LL, in_anchor_LR
    )
    shapes = dali.fn.stack(
        size_UL, size_UR, size_LL, size_LR
    )

    bboxes_UL = bbox_adjust_ltrb(bboxes_UL, cuts_x, cuts_y, zeros_f, zeros_f)
    bboxes_UR = bbox_adjust_ltrb(bboxes_UR, 1.0 - cuts_x, cuts_y, cuts_x, zeros_f)
    bboxes_LL = bbox_adjust_ltrb(bboxes_LL, cuts_x, 1.0 - cuts_y, zeros_f, cuts_y)
    bboxes_LR = bbox_adjust_ltrb(bboxes_LR, 1.0 - cuts_x, 1.0 - cuts_y, cuts_x, cuts_y)
    stacked_bboxes = dali.fn.cat(bboxes_UL, bboxes_UR, bboxes_LL, bboxes_LR)
    stacked_labels = dali.fn.cat(labels_UL, labels_UR, labels_LL, labels_LR)

    mosaic = dali.fn.multi_paste(images, in_ids=idx, output_size=image_size, in_anchors=in_anchors,
                                 shapes=shapes, out_anchors=out_anchors, dtype=dali.types.DALIDataType.UINT8)

    return mosaic, stacked_bboxes, stacked_labels
