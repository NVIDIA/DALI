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
from __future__ import division

import argparse
import os
from math import ceil, sqrt

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

import nvidia.dali.tfrecord as tfrec

from test_utils import compare_pipelines, get_dali_extra_path

test_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco')
test_dummy_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco_dummy')

def coco_anchors():
    anchors = []

    fig_size = 300
    feat_sizes = [38, 19, 10, 5, 3, 1]
    feat_count = len(feat_sizes)
    steps = [8., 16., 32., 64., 100., 300.]
    scales = [21., 45., 99., 153., 207., 261., 315.]
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
            w, h = sizes[0], sizes[1]

            for i in range(feat_sizes[idx]):
                for j in range(feat_sizes[idx]):
                    cx = (j + 0.5) / fks[idx]
                    cy = (i + 0.5) / fks[idx]

                    cx = max(min(cx, 1.), 0.)
                    cy = max(min(cy, 1.), 0.)
                    w = max(min(w, 1.), 0.)
                    h = max(min(h, 1.), 0.)

                    anchors.append(cx - 0.5 * w)
                    anchors.append(cy - 0.5 * h)
                    anchors.append(cx + 0.5 * w)
                    anchors.append(cy + 0.5 * h)

                    anchor_idx = anchor_idx + 1
    return anchors


class TFRecordDetectionPipeline(Pipeline):
    def __init__(self, args):
        super(TFRecordDetectionPipeline, self).__init__(
            args.batch_size, args.num_workers, 0, 0)
        self.input = ops.TFRecordReader(
            path = os.path.join(test_dummy_data_path, 'small_coco.tfrecord'),
            index_path = os.path.join(test_dummy_data_path, 'small_coco_index.idx'),
            features = {
                'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/object/class/label':  tfrec.VarLenFeature([1], tfrec.int64,  0),
                'image/object/bbox':    tfrec.VarLenFeature([4], tfrec.float32, 0.0),
            },
            shard_id=0,
            num_shards=1,
            random_shuffle=False)

        self.decode_gpu = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.cast = ops.Cast(dtype = types.INT32)
        self.box_encoder = ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=coco_anchors())


    def define_graph(self):
        inputs = self.input()
        input_images = inputs["image/encoded"]

        image_gpu = self.decode_gpu(input_images)
        labels = self.cast(inputs['image/object/class/label'])
        encoded_boxes, encoded_labels = self.box_encoder(inputs['image/object/bbox'], labels)

        return (
            image_gpu,
            inputs['image/object/bbox'],
            labels,
            encoded_boxes,
            encoded_labels)


class COCODetectionPipeline(Pipeline):
    def __init__(self, args, data_path = test_data_path):
        super(COCODetectionPipeline, self).__init__(
            args.batch_size, args.num_workers, 0, 0)

        self.input = ops.COCOReader(
            file_root=os.path.join(data_path, 'images'),
            annotations_file=os.path.join(data_path, 'instances.json'),
            shard_id=0,
            num_shards=1,
            ratio=True,
            ltrb=True,
            random_shuffle=False)

        self.decode_gpu = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.box_encoder = ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=coco_anchors())


    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)
        encoded_boxes, encoded_labels = self.box_encoder(boxes, labels)

        return (
            image_gpu,
            boxes,
            labels,
            encoded_boxes,
            encoded_labels)


def print_args(args):
    print('Args values:')
    for arg in vars(args):
        print('{0} = {1}'.format(arg, getattr(args, arg)))
    print()


def run_test(args):
    print_args(args)

    pipe_tf = TFRecordDetectionPipeline(args)
    pipe_coco = COCODetectionPipeline(args, test_dummy_data_path)

    compare_pipelines(pipe_tf, pipe_coco, 1, 64)


def make_parser():
    parser = argparse.ArgumentParser(description='COCO Tfrecord test')
    parser.add_argument(
        '-i', '--iters', default=None, type=int, metavar='N',
        help='number of iterations to run (default: whole dataset)')
    parser.add_argument(
        '-w', '--num_workers', default=4, type=int, metavar='N',
        help='number of worker threads (default: %(default)s)')

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.batch_size = 1

    run_test(args)
