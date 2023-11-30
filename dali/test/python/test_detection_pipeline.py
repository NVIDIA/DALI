# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
from math import ceil, sqrt

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from test_utils import get_dali_extra_path
from test_utils import to_array

test_data_root = get_dali_extra_path()


def coco_anchors():
    anchors = []

    fig_size = 300
    feat_sizes = [38, 19, 10, 5, 3, 1]
    feat_count = len(feat_sizes)
    steps = [8.0, 16.0, 32.0, 64.0, 100.0, 300.0]
    scales = [21.0, 45.0, 99.0, 153.0, 207.0, 261.0, 315.0]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fks = []
    for step in steps:
        fks.append(fig_size / step)

    anchor_idx = 0
    for idx in range(feat_count):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = sqrt(sk1 * sk2)

        all_sizes = [[sk1, sk1], [sk3, sk3]]

        for alpha in aspect_ratios[idx]:
            w = sk1 * sqrt(alpha)
            h = sk1 / sqrt(alpha)
            all_sizes.append([w, h])
            all_sizes.append([h, w])

        for sizes in all_sizes:
            w = sizes[0]
            h = sizes[1]

            for i in range(feat_sizes[idx]):
                for j in range(feat_sizes[idx]):
                    cx = (j + 0.5) / fks[idx]
                    cy = (i + 0.5) / fks[idx]

                    cx = max(min(cx, 1.0), 0.0)
                    cy = max(min(cy, 1.0), 0.0)
                    w = max(min(w, 1.0), 0.0)
                    h = max(min(h, 1.0), 0.0)

                    anchors.append(cx - 0.5 * w)
                    anchors.append(cy - 0.5 * h)
                    anchors.append(cx + 0.5 * w)
                    anchors.append(cy + 0.5 * h)

                    anchor_idx = anchor_idx + 1
    return anchors


def horizontal_flip_ref(image, boxes):
    if len(boxes.shape) == 1:
        boxes[0], boxes[2] = 1.0 - boxes[2], 1.0 - boxes[0]
    else:
        boxes[:, 0], boxes[:, 2] = 1.0 - boxes[:, 2], 1.0 - boxes[:, 0]

    return np.fliplr(image), boxes


def normalize_ref(image):
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    image = image.astype(dtype=np.float64).transpose((2, 0, 1)) / 255
    for plane, (m, s) in zip(range(len(image)), zip(normalization_mean, normalization_std)):
        image[plane] = (image[plane] - m) / s
    return image


def resize_ref(image, size):
    from PIL import Image

    return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))


class DetectionPipeline(Pipeline):
    def __init__(self, args, device_id, file_root, annotations_file):
        super(DetectionPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            prefetch_queue_depth=args.prefetch,
            seed=args.seed,
        )

        # Reading COCO dataset
        self.input = ops.readers.COCO(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=device_id,
            num_shards=args.num_gpus,
            ratio=True,
            ltrb=True,
            random_shuffle=True,
        )

        self.decode_cpu = ops.decoders.Image(device="cpu", output_type=types.RGB)
        self.decode_crop = ops.decoders.ImageSlice(device="cpu", output_type=types.RGB)

        self.decode_gpu = ops.decoders.Image(
            device="mixed", output_type=types.RGB, hw_decoder_load=0
        )
        self.decode_gpu_crop = ops.decoders.ImageSlice(
            device="mixed", output_type=types.RGB, hw_decoder_load=0
        )

        self.ssd_crop = ops.SSDRandomCrop(device="cpu", num_attempts=1, seed=args.seed)
        self.random_bbox_crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            bbox_layout="xyXY",
            seed=args.seed,
        )

        self.slice_cpu = ops.Slice(device="cpu")
        self.slice_gpu = ops.Slice(device="gpu")

        self.resize_cpu = ops.Resize(
            device="cpu",
            resize_x=300,
            resize_y=300,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR,
        )
        self.resize_gpu = ops.Resize(
            device="gpu",
            resize_x=300,
            resize_y=300,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR,
        )

        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        crop_size = (300, 300)
        self.normalize_cpu = ops.CropMirrorNormalize(
            device="cpu", crop=crop_size, mean=mean, std=std, mirror=0, dtype=types.FLOAT
        )
        self.normalize_gpu = ops.CropMirrorNormalize(
            device="gpu", crop=crop_size, mean=mean, std=std, mirror=0, dtype=types.FLOAT
        )

        self.twist_cpu = ops.ColorTwist(device="cpu")
        self.twist_gpu = ops.ColorTwist(device="gpu")

        self.hsv_cpu = ops.Hsv(device="cpu", dtype=types.FLOAT)
        self.hsv_gpu = ops.Hsv(device="gpu", dtype=types.FLOAT)

        self.bc_cpu = ops.BrightnessContrast(device="cpu", dtype=types.UINT8, contrast_center=128)
        self.bc_gpu = ops.BrightnessContrast(device="gpu", dtype=types.UINT8, contrast_center=128)

        self.flip_cpu = ops.Flip(device="cpu")
        self.bbox_flip_cpu = ops.BbFlip(device="cpu", ltrb=True)

        self.flip_gpu = ops.Flip(device="gpu")
        self.bbox_flip_gpu = ops.BbFlip(device="gpu", ltrb=True)

        default_boxes = coco_anchors()
        self.box_encoder_cpu = ops.BoxEncoder(device="cpu", criteria=0.5, anchors=default_boxes)
        self.box_encoder_gpu = ops.BoxEncoder(device="gpu", criteria=0.5, anchors=default_boxes)
        self.box_encoder_cpu_offsets = ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            offset=True,
            scale=2,
            stds=[0.1, 0.1, 0.2, 0.2],
            anchors=default_boxes,
        )
        self.box_encoder_gpu_offsets = ops.BoxEncoder(
            device="gpu",
            criteria=0.5,
            offset=True,
            scale=2,
            stds=[0.1, 0.1, 0.2, 0.2],
            anchors=default_boxes,
        )

        # Random variables
        self.saturation_rng = ops.random.Uniform(range=[0.8, 1.2])
        self.contrast_rng = ops.random.Uniform(range=[0.5, 1.5])
        self.brighness_rng = ops.random.Uniform(range=[0.875, 1.125])
        self.hue_rng = ops.random.Uniform(range=[-45, 45])

    def define_graph(self):
        # Random variables
        saturation = self.saturation_rng()
        contrast = self.contrast_rng()
        brightness = self.brighness_rng()
        hue = self.hue_rng()

        inputs, boxes, labels = self.input(name="Reader")

        image = self.decode_cpu(inputs)
        image_ssd_crop, boxes_ssd_crop, labels_ssd_crop = self.ssd_crop(image, boxes, labels)

        crop_begin, crop_size, boxes_random_crop, labels_random_crop = self.random_bbox_crop(
            boxes, labels
        )
        image_decode_crop = self.decode_crop(inputs, crop_begin, crop_size)

        image_slice_cpu = self.slice_cpu(image, crop_begin, crop_size)
        image_slice_gpu = self.slice_gpu(image.gpu(), crop_begin, crop_size)

        image_resized_cpu = self.resize_cpu(image_ssd_crop)
        image_resized_gpu = self.resize_gpu(image_ssd_crop.gpu())

        image_normalized_cpu = self.normalize_cpu(image_resized_cpu)
        image_normalized_gpu = self.normalize_gpu(image_resized_cpu.gpu())

        image_twisted_cpu = self.hsv_cpu(image_ssd_crop, saturation=saturation, hue=hue)
        image_twisted_cpu = self.bc_cpu(image_twisted_cpu, brightness=brightness, contrast=contrast)
        image_twisted_gpu = self.hsv_gpu(image_ssd_crop.gpu(), saturation=saturation, hue=hue)
        image_twisted_gpu = self.bc_gpu(image_twisted_gpu, brightness=brightness, contrast=contrast)

        image_legacy_twisted_cpu = self.twist_cpu(
            image_ssd_crop, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue
        )
        image_legacy_twisted_gpu = self.twist_gpu(
            image_ssd_crop.gpu(),
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue,
        )

        image_flipped_cpu = self.flip_cpu(image_resized_cpu)
        boxes_flipped_cpu = self.bbox_flip_cpu(boxes_ssd_crop)

        image_flipped_gpu = self.flip_gpu(image_resized_cpu.gpu())
        boxes_flipped_gpu = self.bbox_flip_gpu(boxes_ssd_crop.gpu())

        encoded_boxes_cpu, encoded_labels_cpu = self.box_encoder_cpu(
            boxes_ssd_crop, labels_ssd_crop
        )
        encoded_boxes_gpu, encoded_labels_gpu = self.box_encoder_gpu(
            boxes_ssd_crop.gpu(), labels_ssd_crop.gpu()
        )

        encoded_offset_boxes_cpu, encoded_offset_labels_cpu = self.box_encoder_cpu_offsets(
            boxes_ssd_crop, labels_ssd_crop
        )
        encoded_offset_boxes_gpu, encoded_offset_labels_gpu = self.box_encoder_gpu_offsets(
            boxes_ssd_crop.gpu(), labels_ssd_crop.gpu()
        )

        image_gpu = self.decode_gpu(inputs)
        image_gpu_slice_gpu = self.slice_gpu(image_gpu, crop_begin, crop_size)
        image_decode_crop_gpu = self.decode_gpu_crop(inputs, crop_begin, crop_size)

        return (
            labels,
            image_ssd_crop,
            image_decode_crop,
            image_slice_cpu,
            image_slice_gpu,
            boxes_ssd_crop,
            boxes_random_crop,
            labels_ssd_crop,
            labels_random_crop,
            image_resized_cpu,
            image_resized_gpu,
            image_normalized_cpu,
            image_normalized_gpu,
            image_twisted_cpu,
            image_twisted_gpu,
            image_legacy_twisted_cpu,
            image_legacy_twisted_gpu,
            image_flipped_cpu,
            image_flipped_gpu,
            boxes_flipped_cpu,
            boxes_flipped_gpu,
            encoded_boxes_cpu,
            encoded_boxes_gpu,
            encoded_labels_cpu,
            encoded_labels_gpu,
            encoded_offset_boxes_cpu,
            encoded_offset_boxes_gpu,
            encoded_offset_labels_cpu,
            encoded_offset_labels_gpu,
            image_decode_crop_gpu,
            image_gpu_slice_gpu,
        )


def data_paths(use_full_coco):
    if use_full_coco:
        coco = "/data/coco/coco-2017/coco2017/"
        train = os.path.join(coco, "train2017")
        train_annotations = os.path.join(coco, "annotations/instances_train2017.json")

        val = os.path.join(coco, "val2017")
        val_annotations = os.path.join(coco, "annotations/instances_val2017.json")
        dataset = [(train, train_annotations), (val, val_annotations)]
    else:
        train = os.path.join(test_data_root, "db", "coco", "images")
        train_annotations = os.path.join(test_data_root, "db", "coco", "instances.json")
        dataset = [(train, train_annotations)]

    return dataset


def set_iters(args, dataset_size):
    if args.iters is None:
        args.iters = int(ceil(dataset_size / (args.batch_size * args.num_gpus)))


def compare(val_1, val_2, reference=None):
    test = np.allclose(val_1, val_2)
    if reference is not None:
        test = test and np.allclose(val_1, reference)
        test = test and np.allclose(val_2, reference)

    return test


def crop_border(image, border):
    return image[border:-border, border:-border, :]


def diff_against_eps(image_1, image_2, eps):
    return np.absolute(image_1.astype(float) - image_2.astype(float)).max() <= eps


def relaxed_compare(val_1, val_2, reference=None, eps=1, border=0):
    test = diff_against_eps(val_1, val_2, eps)

    if reference is not None:
        if border != 0:
            reference = crop_border(reference, border)
            test = test and diff_against_eps(reference, crop_border(val_1, border), eps)
            test = test and diff_against_eps(reference, crop_border(val_2, border), eps)
        else:
            test = test and diff_against_eps(reference, val_1, eps)
            test = test and diff_against_eps(reference, val_2, eps)
    return test


def run_for_dataset(args, dataset):
    print("Build pipeline")
    pipes = [
        DetectionPipeline(args, device_id, dataset[0], dataset[1])
        for device_id in range(args.num_gpus)
    ]
    [pipe.build() for pipe in pipes]

    set_iters(args, pipes[0].epoch_size("Reader"))

    for iter in range(args.iters):
        for pipe in pipes:
            (
                labels,
                image_ssd_crop,
                image_decode_crop,
                image_slice_cpu,
                image_slice_gpu,
                boxes_ssd_crop,
                boxes_random_crop,
                labels_ssd_crop,
                labels_random_crop,
                image_resized_cpu,
                image_resized_gpu,
                image_normalized_cpu,
                image_normalized_gpu,
                image_twisted_cpu,
                image_twisted_gpu,
                image_legacy_twisted_cpu,
                image_legacy_twisted_gpu,
                image_flipped_cpu,
                image_flipped_gpu,
                boxes_flipped_cpu,
                boxes_flipped_gpu,
                encoded_boxes_cpu,
                encoded_boxes_gpu,
                encoded_labels_cpu,
                encoded_labels_gpu,
                encoded_offset_boxes_cpu,
                encoded_offset_boxes_gpu,
                encoded_offset_labels_cpu,
                encoded_offset_labels_gpu,
                image_decode_crop_gpu,
                image_gpu_slice_gpu,
            ) = [np.squeeze(to_array(out)) for out in pipe.run()]
            # Check reader
            labels = ((labels > 0) & (labels <= 80)).all()

            # Check cropping ops
            decode_crop = compare(image_ssd_crop, image_decode_crop)
            slice_cpu = compare(image_ssd_crop, image_slice_cpu)
            slice_gpu = compare(image_ssd_crop, image_slice_gpu)
            decode_crop_gpu = compare(image_gpu_slice_gpu, image_decode_crop_gpu)
            image_crop = decode_crop and slice_cpu and slice_gpu and decode_crop_gpu
            boxes_crop = compare(boxes_ssd_crop, boxes_random_crop)
            labels_crop = compare(labels_ssd_crop, labels_random_crop)
            crop = image_crop and boxes_crop and labels_crop

            hsv_bc_twist = relaxed_compare(image_twisted_gpu, image_legacy_twisted_gpu, eps=4)

            # Check resizing ops
            resize = relaxed_compare(
                val_1=image_resized_cpu,
                val_2=image_resized_gpu,
                reference=resize_ref(image_ssd_crop, (300, 300)),
                border=1,
            )

            # Check normalizing ops
            image_normalized_ref = normalize_ref(image_resized_cpu)
            normalize = compare(image_normalized_cpu, image_normalized_gpu, image_normalized_ref)

            # Check twisting ops
            twist_gpu_cpu = relaxed_compare(image_twisted_cpu, image_twisted_gpu, eps=2)
            twist = twist_gpu_cpu and hsv_bc_twist

            # Check flipping ops
            image_flipped_ref, boxes_flipped_ref = horizontal_flip_ref(
                image_resized_cpu, boxes_ssd_crop
            )
            image_flip = compare(image_flipped_cpu, image_flipped_gpu, image_flipped_ref)
            boxes_flip = compare(boxes_flipped_cpu, boxes_flipped_gpu, boxes_flipped_ref)
            flip = image_flip and boxes_flip

            # Check box encoding ops
            encoded_boxes = compare(encoded_boxes_cpu, encoded_boxes_gpu)
            encoded_labels = compare(encoded_labels_cpu, encoded_labels_gpu)
            encoded_boxes_offset = compare(encoded_offset_boxes_cpu, encoded_offset_boxes_gpu)
            encoded_labels_offset = compare(encoded_offset_labels_cpu, encoded_offset_labels_gpu)
            encoded_labels_cpu = compare(encoded_labels_cpu, encoded_offset_labels_cpu)
            encoded_labels_gpu = compare(encoded_labels_gpu, encoded_offset_labels_gpu)
            box_encoder = (
                encoded_boxes
                and encoded_boxes_offset
                and encoded_labels
                and encoded_labels_offset
                and encoded_labels_cpu
                and encoded_labels_gpu
            )

            if (
                not labels
                or not crop
                or not resize
                or not normalize
                or not twist
                or not flip
                or not box_encoder
            ):
                print("Error during iteration", iter)
                print("Labels = ", labels)

                print("Crop = ", crop)
                print("  decode_crop =", decode_crop)
                print("  decode_crop_gpu =", decode_crop_gpu)
                print("  slice_cpu =", slice_cpu)
                print("  slice_gpu =", slice_gpu)
                print("  boxes_crop =", boxes_crop)
                print("  labels_crop =", labels_crop)

                print("Resize =", resize)

                print("Normalize =", normalize)

                print("Twist =", twist)
                print("  twist gpu vs cpu = ", twist_gpu_cpu)
                print("  HSV + BC vs legacy Twist = ", hsv_bc_twist)

                print("Flip =", flip)
                print("  image_flip =", image_flip)
                print("  boxes_flip =", boxes_flip)

                print("Box encoder =", box_encoder)
                print("  encoded_boxes =", encoded_boxes)
                print("  encoded_boxes_offset =", encoded_boxes_offset)
                print("  encoded_labels =", encoded_labels)
                print("  encoded_labels_offset =", encoded_labels_offset)
                print("  encoded_labels_cpu =", encoded_labels_cpu)
                print("  encoded_labels_gpu =", encoded_labels_gpu)

                exit(1)

        if not iter % 100:
            print("Iteration: {}/ {}".format(iter + 1, args.iters))
    print("OK")


def print_args(args):
    print("Args values:")
    for arg in vars(args):
        print("{0} = {1}".format(arg, getattr(args, arg)))
    print()


def run_test(args):
    print_args(args)

    for dataset in data_paths(args.use_full_coco):
        print("Run DetectionPipeline test for", dataset[0])
        run_for_dataset(args, dataset)


def make_parser():
    parser = argparse.ArgumentParser(description="Detection pipeline test")
    parser.add_argument(
        "-i",
        "--iters",
        default=None,
        type=int,
        metavar="N",
        help="number of iterations to run (default: whole dataset)",
    )
    parser.add_argument(
        "-g",
        "--num_gpus",
        default=1,
        type=int,
        metavar="N",
        help="number of GPUs (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        metavar="N",
        help="seed for random ops (default: random seed)",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=3,
        type=int,
        metavar="N",
        help="number of worker threads (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--prefetch",
        default=2,
        type=int,
        metavar="N",
        help="prefetch queue depth (default: %(default)s)",
    )
    parser.add_argument(
        "--use_full_coco", action="store_true", help="Use full COCO data set for this test"
    )

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.batch_size = 1

    run_test(args)
