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

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

import nvidia.dali.tfrecord as tfrec
from test_detection_pipeline import coco_anchors

from test_utils import compare_pipelines, get_dali_extra_path

test_data_path = os.path.join(get_dali_extra_path(), "db", "coco")
test_dummy_data_path = os.path.join(get_dali_extra_path(), "db", "coco_dummy")


class TFRecordDetectionPipeline(Pipeline):
    def __init__(self, args):
        super(TFRecordDetectionPipeline, self).__init__(args.batch_size, args.num_workers, 0, 0)
        self.input = ops.readers.TFRecord(
            path=os.path.join(test_dummy_data_path, "small_coco.tfrecord"),
            index_path=os.path.join(test_dummy_data_path, "small_coco_index.idx"),
            features={
                "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/object/class/label": tfrec.VarLenFeature([], tfrec.int64, 0),
                "image/object/bbox": tfrec.VarLenFeature([4], tfrec.float32, 0.0),
            },
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
        )

        self.decode_gpu = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.cast = ops.Cast(dtype=types.INT32)
        self.box_encoder = ops.BoxEncoder(device="cpu", criteria=0.5, anchors=coco_anchors())

    def define_graph(self):
        inputs = self.input()
        input_images = inputs["image/encoded"]

        image_gpu = self.decode_gpu(input_images)
        labels = self.cast(inputs["image/object/class/label"])
        encoded_boxes, encoded_labels = self.box_encoder(inputs["image/object/bbox"], labels)

        return (image_gpu, inputs["image/object/bbox"], labels, encoded_boxes, encoded_labels)


class COCODetectionPipeline(Pipeline):
    def __init__(self, args, data_path=test_data_path):
        super(COCODetectionPipeline, self).__init__(args.batch_size, args.num_workers, 0, 0)

        self.input = ops.readers.COCO(
            file_root=os.path.join(data_path, "images"),
            annotations_file=os.path.join(data_path, "instances.json"),
            shard_id=0,
            num_shards=1,
            ratio=True,
            ltrb=True,
            random_shuffle=False,
        )

        self.decode_gpu = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.box_encoder = ops.BoxEncoder(device="cpu", criteria=0.5, anchors=coco_anchors())

    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        image_gpu = self.decode_gpu(inputs)
        encoded_boxes, encoded_labels = self.box_encoder(boxes, labels)

        return (image_gpu, boxes, labels, encoded_boxes, encoded_labels)


def print_args(args):
    print("Args values:")
    for arg in vars(args):
        print("{0} = {1}".format(arg, getattr(args, arg)))
    print()


def run_test(args):
    print_args(args)

    pipe_tf = TFRecordDetectionPipeline(args)
    pipe_coco = COCODetectionPipeline(args, test_dummy_data_path)

    compare_pipelines(pipe_tf, pipe_coco, 1, 64)


def make_parser():
    parser = argparse.ArgumentParser(description="COCO Tfrecord test")
    parser.add_argument(
        "-i",
        "--iters",
        default=None,
        type=int,
        metavar="N",
        help="number of iterations to run (default: whole dataset)",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=4,
        type=int,
        metavar="N",
        help="number of worker threads (default: %(default)s)",
    )

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.batch_size = 1

    run_test(args)
