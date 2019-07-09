# # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from __future__ import print_function
# from __future__ import division

# import argparse
# import os
# from math import ceil, sqrt

# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline

# import nvidia.dali.tfrecord as tfrec

# from test_utils import compare_pipelines

# test_data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db', 'coco')


# def coco_anchors():
#     anchors = []

#     fig_size = 300
#     feat_sizes = [38, 19, 10, 5, 3, 1]
#     feat_count = len(feat_sizes)
#     steps = [8., 16., 32., 64., 100., 300.]
#     scales = [21., 45., 99., 153., 207., 261., 315.]
#     aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

#     fks = []
#     for step in steps:
#         fks.append(fig_size / step)

#     anchor_idx = 0
#     for idx in range(feat_count):
#         sk1 = scales[idx] / fig_size
#         sk2 = scales[idx + 1] / fig_size
#         sk3 = sqrt(sk1 * sk2)

#         all_sizes = [[sk1, sk1], [sk3, sk3]]

#         for alpha in aspect_ratios[idx]:
#             w = sk1 * sqrt(alpha)
#             h = sk1 / sqrt(alpha)
#             all_sizes.append([w, h])
#             all_sizes.append([h, w])

#         for sizes in all_sizes:
#             w, h = sizes[0], sizes[1]

#             for i in range(feat_sizes[idx]):
#                 for j in range(feat_sizes[idx]):
#                     cx = (j + 0.5) / fks[idx]
#                     cy = (i + 0.5) / fks[idx]

#                     cx = max(min(cx, 1.), 0.)
#                     cy = max(min(cy, 1.), 0.)
#                     w = max(min(w, 1.), 0.)
#                     h = max(min(h, 1.), 0.)

#                     anchors.append(cx - 0.5 * w)
#                     anchors.append(cy - 0.5 * h)
#                     anchors.append(cx + 0.5 * w)
#                     anchors.append(cy + 0.5 * h)

#                     anchor_idx = anchor_idx + 1
#     return anchors


# class TFRecordDetectionPipeline(Pipeline):
#     def __init__(self, args):
#         super(TFRecordDetectionPipeline, self).__init__(
#             args.batch_size, args.num_workers, 0, 0)
#         self.input = ops.TFRecordReader(
#             path = os.path.join(test_data_path, 'small_coco.tfrecord'), 
#             index_path = os.path.join(test_data_path, 'small_coco_index.idx'),
#             features = {
#                 'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
#                 'image/object/class/label':  tfrec.VarLenFeature([1], tfrec.int64,  0),
#                 'image/object/bbox':    tfrec.VarLenFeature([4], tfrec.float32, 0.0),
#             },
#             shard_id=0,
#             num_shards=1,
#             random_shuffle=False)

#         self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
#         self.cast = ops.Cast(dtype = types.INT32)
#         self.box_encoder = ops.BoxEncoder(
#             device="cpu",
#             criteria=0.5,
#             anchors=coco_anchors())


#     def define_graph(self):
#         inputs = self.input()
#         input_images = inputs["image/encoded"]

#         image_gpu = self.decode_gpu(input_images)
#         labels = self.cast(inputs['image/object/class/label'])
#         encoded_boxes, encoded_labels = self.box_encoder(inputs['image/object/bbox'], labels)

#         return (
#             image_gpu, 
#             inputs['image/object/bbox'],
#             labels,
#             encoded_boxes,
#             encoded_labels)


# class COCODetectionPipeline(Pipeline):
#     def __init__(self, args):
#         super(COCODetectionPipeline, self).__init__(
#             args.batch_size, args.num_workers, 0, 0)

#         self.input = ops.COCOReader(
#             file_root=os.path.join(test_data_path, 'images'),
#             annotations_file=os.path.join(test_data_path, 'instances.json'),
#             shard_id=0,
#             num_shards=1,
#             ratio=True,
#             ltrb=True,
#             random_shuffle=False)

#         self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
#         self.box_encoder = ops.BoxEncoder(
#             device="cpu",
#             criteria=0.5,
#             anchors=coco_anchors())


#     def define_graph(self):
#         inputs, boxes, labels = self.input(name="Reader")
#         image_gpu = self.decode_gpu(inputs)
#         encoded_boxes, encoded_labels = self.box_encoder(boxes, labels)

#         return (
#             image_gpu, 
#             boxes, 
#             labels,
#             encoded_boxes,
#             encoded_labels)


# def print_args(args):
#     print('Args values:')
#     for arg in vars(args):
#         print('{0} = {1}'.format(arg, getattr(args, arg)))
#     print()


# def run_test(args):
#     print_args(args)

#     pipe_tf = TFRecordDetectionPipeline(args)
#     pipe_coco = COCODetectionPipeline(args)

#     compare_pipelines(pipe_tf, pipe_coco, 1, 64)


# def make_parser():
#     parser = argparse.ArgumentParser(description='COCO Tfrecord test')
#     parser.add_argument(
#         '-i', '--iters', default=None, type=int, metavar='N',
#         help='number of iterations to run (default: whole dataset)')
#     parser.add_argument(
#         '-w', '--num_workers', default=4, type=int, metavar='N',
#         help='number of worker threads (default: %(default)s)')

#     return parser


# if __name__ == "__main__":
#     parser = make_parser()
#     args = parser.parse_args()
#     args.batch_size = 1

#     run_test(args)


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
import itertools
import os
import random
from math import ceil, sqrt

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.backend_impl import TensorListGPU
from nvidia.dali.pipeline import Pipeline
from PIL import Image

import tensorflow as tf
import nvidia.dali.tfrecord as tfrec
import time

import struct

from test_utils import compare_pipelines


coco_root_dir = '/data/coco_data/coco/val2017'
coco_annotations = '/data/coco_data/coco/annotations/instances_val2017.json'

tfrecord_dali = [
    "/data/coco_data/dali_1/00000-of-00001.tfrecord"
    ]

tfrecord_idx_dali = [
    "/data/coco_data/dali_1/00000-of-00001.idx"
    ]

# tfrecord_dali = [
#     "/home/awolant/Downloads/coco_data/dali_4/00000-of-00004.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali_4/00001-of-00004.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali_4/00002-of-00004.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali_4/00003-of-00004.tfrecord"
#     ]

# tfrecord_idx_dali = [
#     "/home/awolant/Downloads/coco_data/dali_4/00000-of-00004.idx",
#     "/home/awolant/Downloads/coco_data/dali_4/00001-of-00004.idx",
#     "/home/awolant/Downloads/coco_data/dali_4/00002-of-00004.idx",
#     "/home/awolant/Downloads/coco_data/dali_4/00003-of-00004.idx"
#     ]

# tfrecord_dali = [
#     "/home/awolant/Downloads/coco_data/dali/00000-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00001-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00002-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00003-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00004-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00005-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00006-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00007-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00008-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00009-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00010-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00011-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00012-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00013-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00014-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00015-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00016-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00017-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00018-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00019-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00020-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00021-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00022-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00023-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00024-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00025-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00026-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00027-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00028-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00029-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00030-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/dali/00031-of-00032.tfrecord"
#     ]

# tfrecord_idx_dali = [
#     "/home/awolant/Downloads/coco_data/dali/00000-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00001-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00002-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00003-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00004-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00005-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00006-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00007-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00008-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00009-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00010-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00011-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00012-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00013-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00014-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00015-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00016-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00017-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00018-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00019-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00020-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00021-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00022-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00023-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00024-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00025-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00026-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00027-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00028-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00029-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00030-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/dali/00031-of-00032.idx"
#     ]

# tfrecord = [
#     "/home/awolant/Downloads/coco_data/plain_1/00000-of-00001.tfrecord"
#     ]

# tfrecord_idx = [
#     "/home/awolant/Downloads/coco_data/plain_1/00000-of-00001.idx"
#     ]

# tfrecord = [
#     "/home/awolant/Downloads/coco_data/plain_4/00000-of-00004.tfrecord",
    # "/home/awolant/Downloads/coco_data/plain_4/00001-of-00004.tfrecord",
    # "/home/awolant/Downloads/coco_data/plain_4/00002-of-00004.tfrecord",
    # "/home/awolant/Downloads/coco_data/plain_4/00003-of-00004.tfrecord"
    # ]

# tfrecord_idx = [
    # "/home/awolant/Downloads/coco_data/plain_4/00000-of-00004.idx",
    # "/home/awolant/Downloads/coco_data/plain_4/00001-of-00004.idx",
    # "/home/awolant/Downloads/coco_data/plain_4/00002-of-00004.idx",
    # "/home/awolant/Downloads/coco_data/plain_4/00003-of-00004.idx"
    # ]

# tfrecord = [
#     "/home/awolant/Downloads/coco_data/plain_test/00000-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00001-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00002-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00003-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00004-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00005-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00006-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00007-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00008-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00009-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00010-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00011-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00012-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00013-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00014-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00015-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00016-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00017-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00018-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00019-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00020-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00021-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00022-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00023-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00024-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00025-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00026-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00027-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00028-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00029-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00030-of-00032.tfrecord",
#     "/home/awolant/Downloads/coco_data/plain_test/00031-of-00032.tfrecord"
#     ]

# tfrecord_idx = [
#     "/home/awolant/Downloads/coco_data/plain_test/00000-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00001-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00002-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00003-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00004-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00005-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00006-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00007-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00008-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00009-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00010-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00011-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00012-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00013-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00014-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00015-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00016-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00017-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00018-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00019-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00020-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00021-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00022-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00023-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00024-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00025-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00026-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00027-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00028-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00029-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00030-of-00032.idx",
#     "/home/awolant/Downloads/coco_data/plain_test/00031-of-00032.idx"
#     ]

tfrecord = [
    "/data/coco_data/plain_test/00000-of-00001.tfrecord"
    ]

tfrecord_idx = [
    "/data/coco_data/plain_test/00000-of-00001.idx"
    ]


class TFRecordDaliDetectionPipeline(Pipeline):
    def __init__(self, args, device_id):
        super(TFRecordDaliDetectionPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        self.input = ops.TFRecordReader(
            path = tfrecord_dali, 
            index_path = tfrecord_idx_dali,
            features = {
                'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/object/class/label':  tfrec.VarLenFeature([1], tfrec.int64,  0),
                'image/object/bbox':    tfrec.VarLenFeature([4], tfrec.float32, 0.0),
            },
            # features = {
            #     'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
            #     'image/object/class/label':  tfrec.VarLenFeature(tfrec.int64,  0),
            #     'image/object/bbox':    tfrec.VarLenFeature(tfrec.float32, 0.0),
            # },
            shard_id=0,
            num_shards=1,
            random_shuffle=False)

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        # self.cast = ops.Cast(dtype = types.INT32)

    def define_graph(self):
        inputs = self.input()
        input_images = inputs["image/encoded"]

        image_gpu = self.decode_gpu(input_images)
        # labels = self.cast(inputs['image/object/class/label'])

        return (
            image_gpu, 
            inputs['image/object/bbox'],
            # labels
            inputs['image/object/class/label']
            )


class TFRecordDetectionPipeline(Pipeline):
    def __init__(self, args, device_id):
        super(TFRecordDetectionPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        self.input = ops.TFRecordReader(
            path = tfrecord, 
            index_path = tfrecord_idx,
            features = {
                'image/encoded' : tfrec.FixedLenFeature((), tfrec.string, ""),
                'idx': tfrec.FixedLenFeature((), tfrec.int64,  0),
                # 'image/object/class/label':  tfrec.VarLenFeature(tfrec.int64,  0),
                # 'image/object/bbox/xmin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                # 'image/object/bbox/ymin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                # 'image/object/bbox/xmax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                # 'image/object/bbox/ymax':    tfrec.VarLenFeature(tfrec.float32, 0.0)
            },
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
            meta_files_path='/data/coco_data/plain_test/')

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        # self.cast = ops.Cast(dtype = types.INT32)

    def define_graph(self):
        inputs = self.input()
        input_images = inputs["image/encoded"]

        # print(inputs)

        image_gpu = self.decode_gpu(input_images)
        # labels = self.cast(inputs['image/object/class/label'])

        return (
            image_gpu,
            # inputs["idx"],
            inputs["boxes"],
            inputs["labels"]
            # inputs['image/object/bbox/xmin'],
            # inputs['image/object/bbox/ymin'],
            # inputs['image/object/bbox/xmax'],
            # inputs['image/object/bbox/ymax'],
            # labels
            # inputs['image/object/class/label']
            )


class COCODetectionPipeline(Pipeline):
    def __init__(self, args, device_id):
        super(COCODetectionPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        self.input = ops.COCOReader(
            file_root=coco_root_dir,
            annotations_file=coco_annotations,
            ratio=True,
            ltrb=True,
            shard_id=0,
            num_shards=1,
            skip_empty=True,
            random_shuffle=True)

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)


    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)

        return (image_gpu, boxes, labels)

class FastCocoDetectionPipeline(Pipeline):
    def __init__(self, args, device_id):
        super(FastCocoDetectionPipeline, self).__init__(
            batch_size=args.batch_size,
            num_threads=args.num_workers,
            device_id=device_id,
            seed=args.seed)

        self.input = ops.FastCocoReader(
            file_root='/data/coco_data/coco/val2017',
            random_shuffle=True,
            shard_id=0,
            num_shards=1,
            meta_files_path='/data/coco_data/coco_fast/')

        self.decode_gpu = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)


    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)

        return (image_gpu, boxes, labels)


def print_args(args):
    print('Args values:')
    for arg in vars(args):
        print('{0} = {1}'.format(arg, getattr(args, arg)))
    print()


def run_test(args, pipe_func):
    print_args(args)

    times = []

    for run in range(args.reps):
        print('Run ', run)

        pipe = pipe_func(args, 0)
        pipe.build()

        start = time.time()
        for iter in range(args.iters):
            out = pipe.run()
            # print(out[1].as_array())
            # print(out[1].as_array().shape)
            # print(out[2].as_array())
            # print(out[2].as_array().shape)
        end = time.time()

        times.append(args.iters * args.batch_size / (end-start))
    print(times)
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

    # print('Warmup')
    # run_test(args, COCODetectionPipeline)

    # print('COCO reader')
    # run_test(args, COCODetectionPipeline)

    # print('Fast COCO reader')
    # run_test(args, FastCocoDetectionPipeline)

    # print('TFRecord')
    # run_test(args, TFRecordDetectionPipeline)

    # print('TFRecord DALI')
    # run_test(args, TFRecordDaliDetectionPipeline)

    # pipe_1 = COCODetectionPipeline(args, 0)
    # pipe_2 = TFRecordDetectionPipeline(args, 0)

    pipe_1 = COCODetectionPipeline(args, 0)
    pipe_2 = FastCocoDetectionPipeline(args, 0)

    compare_pipelines(pipe_1, pipe_2, args.batch_size, args.iters)

