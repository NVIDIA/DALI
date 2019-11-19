# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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
import json
import os

from paddle import fluid

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

from tsm import TSM
from utils import load_weights

PRETRAIN_WEIGHTS = 'https://paddlemodels.bj.bcebos.com/video_classification/TSM_final.pdparams'


class VideoPipe(Pipeline):
    def __init__(self, video_files, sequence_length=8, target_size=224,
                 stride=30):
        super(VideoPipe, self).__init__(1, 4, 0, seed=42)
        self.input = ops.VideoReader(
            device="gpu", filenames=video_files,
            sequence_length=sequence_length, stride=stride,
            shard_id=0, num_shards=1, random_shuffle=False)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NFCHW,
            crop=(target_size, target_size),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        images = self.input(name="Reader")
        return self.cmnp(images)


def build(seg_num=8, target_size=224):
    image_shape = [seg_num, 3, target_size, target_size]

    image = fluid.layers.data(
        name='image', shape=image_shape, dtype='float32')

    model = TSM()
    return model(image)


def main():
    seg_num = 8
    target_size = 224

    video_files = [FLAGS.data + '/' + f for f in os.listdir(FLAGS.data)]
    pipeline = VideoPipe(video_files, seg_num, target_size, FLAGS.stride)

    video_loader = DALIGenericIterator(
        pipeline, ['image'], len(video_files), dynamic_shape=True)

    exe = fluid.Executor(fluid.CUDAPlace(0))
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()

    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            fetch_list = build(seg_num, target_size)

    exe.run(startup_prog)
    compiled_eval_prog = fluid.CompiledProgram(eval_prog)

    load_weights(exe, eval_prog, PRETRAIN_WEIGHTS)

    labels = json.load(open("kinetics_labels.json"))

    for idx, batch in enumerate(video_loader):
        fetches = exe.run(
            compiled_eval_prog, feed=batch, fetch_list=fetch_list)
        pred = fetches[0][0]
        topk_indices = pred.argsort()[0 - FLAGS.topk:]
        topk_labels = [labels[i] for i in topk_indices]
        filename = video_files[idx]
        print("prediction for {} is: {}".format(filename, topk_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paddle Temporal Shift Module Inference')
    parser.add_argument('data', metavar='DIR', help='Path to video files')
    parser.add_argument('--topk', '-k', default=1, type=int,
                        metavar='K', help='Top k results (default: 1)')
    parser.add_argument('--stride', '-s', default=30, type=int, metavar='S',
                        help='Distance between frames (default: 30)')
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
