from __future__ import print_function
from __future__ import division

import argparse
import itertools
import os
import random
from math import ceil, sqrt

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


import struct

from test_utils import compare_pipelines

coco_root_dir = '/data/coco_data/coco/val2017'
coco_annotations = '/data/coco_data/coco/annotations/instances_val2017.json'
file_list = "/data/coco_data/file_list.txt"


class TestPipeline(Pipeline):
    def __init__(self, args, device_id, fast):
        super(TestPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        if fast:
            self.input = ops.FastCocoReader(
                file_root=coco_root_dir,
                random_shuffle=True,
                shard_id=0,
                num_shards=1,
                ratio=args.ratio,
                ltrb=args.ltrb,
                skip_empty=args.skip_empty,
                size_threshold=args.size_threshold,
                # meta_files_path='/data/coco_data/coco_fast/',
                # meta_files_path='/data/coco_data/',
                # file_list=file_list,
                annotations_file=coco_annotations,
                dump_meta_files=True,
                dump_meta_files_path='/data/coco_data/'
                # save_img_ids=True,
                )
        else:
            self.input = ops.COCOReader(
            file_root=coco_root_dir,
            random_shuffle=True,
            shard_id=0,
            num_shards=1,
            ratio=args.ratio,
            ltrb=args.ltrb,
            skip_empty=args.skip_empty,
            size_threshold=args.size_threshold,
            annotations_file=coco_annotations,
            # save_img_ids=True,
            )

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)


    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)

        return image_gpu, boxes, labels

class MetaPipeline(Pipeline):
    def __init__(self, args, device_id):
        super(MetaPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        self.input = ops.FastCocoReader(
            file_root=coco_root_dir,
            random_shuffle=True,
            shard_id=0,
            num_shards=1,
            # ratio=args.ratio,
            # ltrb=args.ltrb,
            # skip_empty=args.skip_empty,
            # size_threshold=args.size_threshold,
            # meta_files_path='/data/coco_data/coco_fast/',
            meta_files_path='/data/coco_data/',
            # file_list=file_list,
            # annotations_file=coco_annotations,
            # dump_meta_files=True,
            # dump_meta_files_path='/data/coco_data/'
            # save_img_ids=True,
            )

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)


    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)

        return image_gpu, boxes, labels


def print_args(args):
    print('Args values:')
    for arg in vars(args):
        print('{0} = {1}'.format(arg, getattr(args, arg)))
    print()


def make_parser():
    parser = argparse.ArgumentParser(description='Detection pipeline test')
    parser.add_argument(
        '-i', '--iters', default=1000, type=int, metavar='N',
        help='number of iterations to run')
    parser.add_argument(
        '-b', '--batch-size', default=64, type=int, metavar='N',
        help='batch size')
    parser.add_argument(
        '-s', '--seed', default=0, type=int, metavar='N',
        help='seed for random ops (default: random seed)')
    parser.add_argument(
        '-w', '--num-workers', default=4, type=int, metavar='N',
        help='number of worker threads (default: %(default)s)')
    parser.add_argument(
        '-r', '--reps', default=5, type=int, metavar='N',
        help='number of experiment runs')

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    ratios = [True, False]
    ltrbs = [True, False]
    skip_emptys = [True, False]
    size_thresholds = [0.1, 300.]

    for params in itertools.product(ratios, ltrbs, skip_emptys, size_thresholds):

        args.ratio = params[0]
        args.ltrb = params[1]
        args.skip_empty = params[2]
        args.size_threshold = params[3]

        fast_pipe = TestPipeline(args, 0, True)
        coco_pipe = TestPipeline(args, 0, False)
        compare_pipelines(fast_pipe, coco_pipe, args.batch_size, args.iters)

        coco_pipe = TestPipeline(args, 0, False)
        meta_pipe = MetaPipeline(args, 0)
        compare_pipelines(meta_pipe, coco_pipe, args.batch_size, args.iters)

