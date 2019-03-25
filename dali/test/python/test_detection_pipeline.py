# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.backend_impl import TensorListGPU
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import argparse
import numpy as np
import random
import os
import math


# N = args.gpus             # number of GPUs
# BATCH_SIZE = 1
# LOG_INTERVAL = 100
# WORKERS = 3
# PREFETCH = 2

# for pipe_name in test_data.keys():
#     data_set_len = len(test_data[pipe_name])
#     for i, data_set in enumerate(test_data[pipe_name]):
#         seed = int(random.random() * (1<<64))
#         print ("Seed:", seed)
#         print("Build pipeline")
#         pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=n,
#                            num_gpus=N, data_paths=data_set, prefetch=PREFETCH, seed=1) for n in range(N)]
#         [pipe.build() for pipe in pipes]
#         if args.iters < 0:
#           iters = pipes[0].epoch_size("Reader")
#           iters_tmp = iters
#           iters = iters // BATCH_SIZE
#           if iters_tmp != iters * BATCH_SIZE:
#               iters += 1
#           iters_tmp = iters

#           iters = iters // N
#           if iters_tmp != iters * N:
#               iters += 1
#         else:
#           iters = args.iters

#         print ("RUN {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
#         print (data_set)
#         for j in range(iters):
#             for pipe in pipes:
#                 bboxes_1, labels_1, images_1, bboxes_2, labels_2, images_2, images_3, images_4, \
#                     images_flipped_cpu, boxes_flipped_cpu, images_flipped_gpu, boxes_flipped_gpu = pipe.run()
#                 bboxes_1_arr = np.squeeze(bboxes_1.as_array())
#                 bboxes_2_arr = np.squeeze(bboxes_2.as_array())
#                 labels_1_arr = np.squeeze(labels_1.as_array())
#                 labels_2_arr = np.squeeze(labels_2.as_array())
#                 images_1_arr = images_1.as_array()
#                 images_2_arr = images_2.as_array()
#                 images_3_arr = images_3.as_array()
#                 images_4_arr = images_4.asCPU().as_array()

#                 images_flipped_cpu_arr = images_flipped_cpu.as_array()
#                 boxes_flipped_cpu_arr = np.squeeze(boxes_flipped_cpu.as_array())
#                 images_flipped_gpu_arr = images_flipped_gpu.asCPU().as_array()
#                 boxes_flipped_gpu_arr = np.squeeze(boxes_flipped_gpu.asCPU().as_array())

#                 res = np.allclose(labels_1_arr, labels_2_arr)
#                 if not res:
#                     print(labels_1_arr, "\nvs\n", labels_2_arr)
#                 res_bb = np.allclose(bboxes_1_arr, bboxes_2_arr)
#                 if not res_bb:
#                     print(bboxes_1_arr, "\nvs\n", bboxes_2_arr)
#                 res_img = np.allclose(images_1_arr, images_2_arr) and np.allclose(images_1_arr, images_3_arr) and np.allclose(images_1_arr, images_4_arr)
#                 if not res_img:
#                     print(images_1_arr, "\nvs\n", images_2_arr)
#                     print(images_1_arr, "\nvs\n", images_3_arr)
#                     print(images_1_arr, "\nvs\n", images_4_arr)

#                 res_flip = np.allclose(images_flipped_cpu_arr, images_flipped_gpu_arr)
#                 if not res_bb or not res or not res_img or not res_flip:
#                     print("Labels == ", res)
#                     print("Bboxes == ", res_bb)
#                     print("Images == ", res_img)
#                     print("Flip == ", res_flip)
#                     exit(1)
#             if not j % LOG_INTERVAL:
#                 print("{} {}/ {}".format(pipe_name.__name__, j + 1, iters))

#         print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))


class DetectionPipeline(Pipeline):
    def __init__(self, args, device_id, file_root, annotations_file):
        super(DetectionPipeline, self).__init__(
            args.batch_size, args.num_workers, device_id, args.prefetch, args.seed)

        # Reading COCO dataset
        self.input = ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=device_id,
            num_shards=args.num_gpus,
            ratio=True,
            ltrb=True,
            random_shuffle=True)

        self.decode_cpu = ops.HostDecoder(device="cpu", output_type=types.RGB)
        self.decode_crop = ops.HostDecoderSlice(
            device="cpu", output_type=types.RGB)

        self.ssd_crop = ops.SSDRandomCrop(
            device="cpu", num_attempts=1, seed=args.seed)
        self.random_bbox_crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            seed=args.seed)

        self.slice_cpu = ops.Slice(device="cpu")
        self.slice_gpu = ops.Slice(device="gpu")

        self.flip_cpu = ops.Flip(device="cpu")
        self.bbox_flip_cpu = ops.BbFlip(device="cpu", ltrb=True)

        self.flip_gpu = ops.Flip(device="gpu")
        self.bbox_flip_gpu = ops.BbFlip(device="gpu", ltrb=True)

    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")

        images = self.decode_cpu(inputs)
        images_ssd_crop, boxes_ssd_crop, labels_ssd_crop = self.ssd_crop(
            images, boxes, labels)

        crop_begin, crop_size, boxes_random_crop, labels_random_crop = \
            self.random_bbox_crop(boxes, labels)
        images_decode_crop = self.decode_crop(inputs, crop_begin, crop_size)

        images_slice_cpu = self.slice_cpu(images, crop_begin, crop_size)
        images_slice_gpu = self.slice_gpu(images.gpu(), crop_begin, crop_size)

        images_flipped_cpu = self.flip_cpu(images_ssd_crop)
        boxes_flipped_cpu = self.bbox_flip_cpu(boxes_ssd_crop)

        images_flipped_gpu = self.flip_gpu(images_ssd_crop.gpu())
        boxes_flipped_gpu = self.bbox_flip_gpu(boxes_ssd_crop.gpu())

        return (
            images_ssd_crop, images_decode_crop,
            images_slice_cpu, images_slice_gpu,
            boxes_ssd_crop, boxes_random_crop,
            labels_ssd_crop, labels_random_crop,
            images_flipped_cpu, images_flipped_gpu,
            boxes_flipped_cpu, boxes_flipped_gpu,
        )


def data_paths():
    coco = '/data/coco/coco-2017/coco2017/'
    train = os.path.join(coco, 'train2017')
    train_annotations = os.path.join(
        coco, 'annotations/instances_train2017.json')

    val = os.path.join(coco, 'val2017')
    val_annotations = os.path.join(
        coco, 'annotations/instances_val2017.json')

    return [(train, train_annotations), (val, val_annotations)]


def set_iters(args, dataset_size):
    if args.iters is None:
        args.iters = math.ceil(
            dataset_size / (args.batch_size * args.num_gpus))


def to_array(dali_out):
    if isinstance(dali_out, TensorListGPU):
        dali_out = dali_out.asCPU()

    return np.squeeze(dali_out.as_array())


def compare(val_1, val_2, reference=None):
    test = np.allclose(val_1, val_2)
    if reference is not None:
        test = test and np.allclose(val_1, reference)
        test = test and np.allclose(val_2, reference)

    return test


def run_for_dataset(args, dataset):
    print("Build pipeline")
    pipes = [DetectionPipeline(args, device_id, dataset[0], dataset[1])
             for device_id in range(args.num_gpus)]
    [pipe.build() for pipe in pipes]

    set_iters(args, pipes[0].epoch_size('Reader'))

    for iter in range(args.iters):
        for pipe in pipes:
            images_ssd_crop, images_decode_crop, \
                images_slice_cpu, images_slice_gpu, \
                boxes_ssd_crop, boxes_random_crop, \
                labels_ssd_crop, labels_random_crop,\
                images_flipped_cpu, images_flipped_gpu,\
                boxes_flipped_cpu, boxes_flipped_gpu = \
                [to_array(out) for out in pipe.run()]

            # Check cropping ops
            decode_crop = compare(images_ssd_crop, images_decode_crop)
            slice_cpu = compare(images_ssd_crop, images_slice_cpu)
            slice_gpu = compare(images_ssd_crop, images_slice_gpu)
            images_crop = decode_crop and slice_cpu and slice_gpu
            boxes_crop = compare(boxes_ssd_crop, boxes_random_crop)
            labels_crop = compare(labels_ssd_crop, labels_random_crop)
            crop = images_crop and boxes_crop and labels_crop

            # Check flipping ops
            images_flip = compare(images_flipped_cpu, images_flipped_gpu)
            boxes_flip = compare(boxes_flipped_cpu, boxes_flipped_gpu)
            flip = images_flip and boxes_flip

            if not crop or not flip:
                print('Error during iteration', iter)
                print('Crop = ', crop)
                print('  decode_crop =', decode_crop)
                print('  slice_cpu =', slice_cpu)
                print('  slice_gpu =', slice_gpu)
                print('  boxes_crop =', decode_crop)
                print('  labels_cpu =', labels_crop)
                
                print('Flip =', flip)
                print('  images_flip =', images_flip)
                print('  boxes_flip =', boxes_flip)

                exit(1)

        if not iter % 100:
            print("Iteration: {}/ {}".format(iter + 1, args.iters))

    print()


def print_args(args):
    print('Args values:')
    for arg in vars(args):
        print('{0} = {1}'.format(arg, getattr(args, arg)))
    print()


def run_test(args):
    print_args(args)

    for dataset in data_paths():
        print('Run DetectionPipeline test for', dataset[0])
        run_for_dataset(args, dataset)


def random_seed():
    return int(random.random() * (1 << 32))


def make_parser():
    parser = argparse.ArgumentParser(description='Detection pipeline test')
    parser.add_argument(
        '-i', '--iters', default=None, type=int, metavar='N',
        help='number of iterations to run (default: whole dataset)')
    parser.add_argument(
        '-g', '--num_gpus', default=1, type=int, metavar='N',
        help='number of GPUs (default: %(default)s)')
    parser.add_argument(
        '-s', '--seed', default=random_seed(), type=int, metavar='N',
        help='seed for random ops (default: random seed)')
    parser.add_argument(
        '-b', '--batch_size', default=1, type=int, metavar='N',
        help='batch size (default: %(default)s)')
    parser.add_argument(
        '-w', '--num_workers', default=3, type=int, metavar='N',
        help='number of worker threads (default: %(default)s)')
    parser.add_argument(
        '-p', '--prefetch', default=2, type=int, metavar='N',
        help='prefetch queue depth (default: %(default)s)')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    run_test(args)
