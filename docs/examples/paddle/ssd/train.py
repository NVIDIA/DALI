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
import os
import shutil
import time

import numpy as np

from paddle import fluid

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator

from ssd import SSD
from utils import load_weights

PRETRAIN_WEIGHTS = 'https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_caffe_pretrained.tar'


class HybridTrainPipe(Pipeline):
    def __init__(self,
                 file_root,
                 annotations_file,
                 batch_size=1,
                 device_id=0,
                 num_threads=4,
                 local_rank=0,
                 world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads,
                                              device_id,
                                              seed=42 + device_id)
        self.reader = ops.COCOReader(
            file_root=file_root,
            annotations_file=annotations_file,
            skip_empty=True,
            shard_id=local_rank,
            num_shards=world_size,
            ratio=True,
            ltrb=True,
            shuffle_after_epoch=True)

        self.crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=50)
        self.bbflip = ops.BbFlip(device="cpu", ltrb=True)

        self.roi_decode = ops.ImageDecoderSlice(device="mixed")
        self.resize = ops.Resize(
            device="gpu",
            resize_x=300,
            resize_y=300,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR)
        self.hsv = ops.Hsv(device="gpu", dtype=types.FLOAT)  # use float to avoid clipping and
                                                             # quantizing the intermediate result
        self.bc = ops.BrightnessContrast(device="gpu",
                        contrast_center=128,  # input is in float, but in 0..255 range
                        dtype=types.UINT8)

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            mean=[104., 117., 123.],
            std=[1., 1., 1.],
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            pad_output=False)

        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])
        self.coin = ops.CoinFlip(probability=0.5)
        self.build()

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        flip = self.coin()

        images, bboxes, labels = self.reader()

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        bboxes = self.bbflip(bboxes, horizontal=flip)

        images = self.roi_decode(images, crop_begin, crop_size)
        images = self.resize(images)
        images = self.hsv(images.gpu(), hue=hue, saturation=saturation)
        images = self.bc(images, brightness=brightness, contrast=contrast)
        images = self.cmnp(images, mirror=flip)
        return images, bboxes, labels


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build():
    model = SSD()

    image = fluid.layers.data(
        name='image', shape=[3, 300, 300], dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)

    return model(image, gt_box, gt_label)


def main():
    places = []
    for p in fluid.framework.cuda_places():
        place = fluid.core.Place()
        place.set_place(p)
        places.append(place)

    file_root = os.path.join(FLAGS.data, 'train2017')
    annotations_file = os.path.join(
        FLAGS.data, 'annotations/instances_train2017.json')
    world_size = len(places)

    pipelines = [
        HybridTrainPipe(
            file_root, annotations_file, FLAGS.batch_size, p.gpu_device_id(),
            FLAGS.num_threads, local_rank=idx, world_size=world_size)
        for idx, p in enumerate(places)]

    sample_per_shard = 118287 // world_size
    train_loader = DALIGenericIterator(
        pipelines, ['image', ('gt_box', 1), ('gt_label', 1)],
        sample_per_shard, auto_reset=True, dynamic_shape=True)

    FLAGS.whole_batch_size = FLAGS.batch_size * world_size
    total_steps = 400000
    if FLAGS.check_loss_steps > 0:
        total_steps = FLAGS.check_loss_steps
    milestones = [280000, 360000]
    values = [FLAGS.lr * (0.1**i) for i in range(len(milestones) + 1)]

    exe = fluid.Executor(fluid.CUDAPlace(0))
    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_fetch_list = build()
            learning_rate = fluid.layers.piecewise_decay(
                boundaries=milestones, values=values)
            learning_rate = fluid.layers.linear_lr_warmup(
                learning_rate=learning_rate,
                warmup_steps=500,
                start_lr=FLAGS.lr / 3,
                end_lr=FLAGS.lr)
            decay = FLAGS.weight_decay
            optimizer = fluid.optimizer.Momentum(
                momentum=FLAGS.momentum,
                learning_rate=learning_rate,
                regularization=fluid.regularizer.L2Decay(decay))
            avg_loss = train_fetch_list[0]
            optimizer.minimize(avg_loss)

    exe.run(startup_prog)
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_loss.name)

    load_weights(exe, train_prog, PRETRAIN_WEIGHTS)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    def forever():
        while True:
            try:
                yield next(train_loader)
            except StopIteration:
                pass

    for idx, batch in enumerate(forever()):
        if idx > total_steps:
            break
        data_time.update(time.time() - end)

        fetches = exe.run(
            compiled_train_prog, feed=batch, fetch_list=train_fetch_list)
        loss = np.mean(fetches[0])

        losses.update(loss, FLAGS.whole_batch_size)

        if FLAGS.check_loss_steps > 0:
            if idx == 0:
                loss_start = loss
            else:
                loss_end = loss

        if idx % FLAGS.print_freq == 0 and idx > 1:
            print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      idx, total_steps,
                      FLAGS.whole_batch_size / batch_time.val,
                      FLAGS.whole_batch_size / batch_time.avg,
                      batch_time=batch_time,
                      data_time=data_time, loss=losses))

        if idx % FLAGS.ckpt_freq == 0 and idx > 1:
            ckpt_path = os.path.join('checkpoint', "{:02d}".format(idx))
            if os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path)

            print('Save model to {}.'.format(ckpt_path))
            fluid.io.save_persistables(exe, ckpt_path, train_prog)

        batch_time.update(time.time() - end)
        end = time.time()

    if FLAGS.check_loss_steps > 0:
        assert loss_start > loss_end, \
            'loss should decrease after training for {} steps'.format(
                FLAGS.check_loss_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paddle Single Shot MultiBox Detector Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-j', '--num_threads', default=4, type=int,
                        metavar='N', help='number of threads (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt-freq', '-c', default=5000, type=int,
                        metavar='N',
                        help='checkpoint frequency (default: 5000)')
    parser.add_argument('--check-loss-steps', '-t', default=-1, type=int,
                        metavar='N', help='check N steps for loss convergence')
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
