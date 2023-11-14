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

import argparse
import json
import os

import paddle
import paddle.static as static

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.paddle import DALIGenericIterator

from tsm import TSM
from utils import load_weights

PRETRAIN_WEIGHTS = 'https://paddlemodels.bj.bcebos.com/video_classification/TSM_final.pdparams'


def create_video_pipe(video_files, sequence_length=8, target_size=224,stride=30):
    pipeline = Pipeline(1, 4, 0, seed=42)
    with pipeline:
        images = fn.readers.video(device="gpu", filenames=video_files,
                                  sequence_length=sequence_length, stride=stride,
                                  shard_id=0, num_shards=1, random_shuffle=False,
                                  pad_last_batch=True, name="Reader")
        images = fn.crop_mirror_normalize(images,
                                          dtype=types.FLOAT,
                                          output_layout="FCHW",
                                          crop=(target_size, target_size),
                                          mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        pipeline.set_outputs(images)
    return pipeline


def build(seg_num=8, target_size=224):
    image_shape = [1, seg_num, 3, target_size, target_size]

    image = static.data(name='image', shape=image_shape, dtype='float32')

    model = TSM()
    return model(image)


def main():
    seg_num = 8
    target_size = 224

    video_files = [FLAGS.data + '/' + f for f in os.listdir(FLAGS.data)]
    pipeline = create_video_pipe(video_files, seg_num, target_size, FLAGS.stride)

    video_loader = DALIGenericIterator(
        pipeline, ['image'], reader_name="Reader", dynamic_shape=True)

    exe = static.Executor(paddle.CUDAPlace(0))
    startup_prog = static.Program()
    eval_prog = static.Program()

    with static.program_guard(eval_prog, startup_prog):
        with paddle.utils.unique_name.guard():
            fetch_list = build(seg_num, target_size)

    exe.run(startup_prog)
    compiled_eval_prog = static.CompiledProgram(eval_prog)

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
    # In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode.
    # So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.
    paddle.enable_static()
    main()
