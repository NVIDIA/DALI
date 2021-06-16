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
import math
import os
import shutil
import time

import numpy as np

from paddle import fluid
import paddle

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.paddle import DALIClassificationIterator, LastBatchPolicy


def create_dali_pipeline(batch_size, num_threads, device_id, data_dir, crop, size,
                         shard_id, num_shards, dali_cpu=False, is_training=True):
    pipeline = Pipeline(batch_size, num_threads, device_id, seed=12 + device_id)
    with pipeline:
        images, labels = fn.readers.file(file_root=data_dir,
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         random_shuffle=is_training,
                                         pad_last_batch=True,
                                         name="Reader")
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
        preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
        if is_training:
            images = fn.decoders.image_random_crop(images,
                                                   device=decoder_device, output_type=types.RGB,
                                                   device_memory_padding=device_memory_padding,
                                                   host_memory_padding=host_memory_padding,
                                                   preallocate_width_hint=preallocate_width_hint,
                                                   preallocate_height_hint=preallocate_height_hint,
                                                   random_aspect_ratio=[0.8, 1.25],
                                                   random_area=[0.1, 1.0],
                                                   num_attempts=100)
            images = fn.resize(images,
                               device=dali_device,
                               resize_x=crop,
                               resize_y=crop,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images,
                                       device=decoder_device,
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device=dali_device,
                               size=size,
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(crop, crop),
                                          mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                          std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                          mirror=mirror)
        labels = labels.gpu()
        labels = fn.cast(labels, dtype=types.INT64)
        pipeline.set_outputs(images, labels)
    return pipeline

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
    from resnet import ResNet
    model = ResNet(FLAGS.depth, num_classes=1000)

    image = fluid.layers.data(name='data', shape=[3, 224, 224],
                              dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int32')

    logits = model(image)
    loss, pred = fluid.layers.softmax_with_cross_entropy(
        logits, label, return_softmax=True)
    avg_loss = fluid.layers.mean(x=loss)
    avg_loss.persistable = True
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    return avg_loss, acc_top1, acc_top5


def run(exe, prog, fetch_list, loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    total_batches = int(loader._size / FLAGS.batch_size)

    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)

        loss, prec1, prec5 = exe.run(
            prog, feed=batch, fetch_list=fetch_list)
        prec5 = np.mean(prec5)
        loss = np.mean(loss)
        prec1 = np.mean(prec1)
        prec5 = np.mean(prec5)

        num_items = batch[0]['label'].shape()[0]

        losses.update(loss, num_items)
        top1.update(prec1, num_items)
        top5.update(prec5, num_items)
        batch_time.update(time.time() - end)
        end = time.time()

        if FLAGS.local_rank == 0 and i % FLAGS.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, total_batches,
                      FLAGS.whole_batch_size / batch_time.val,
                      FLAGS.whole_batch_size / batch_time.avg,
                      batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    return batch_time.avg, top1.avg, top5.avg


def main():
    env = os.environ
    FLAGS.local_rank = int(env.get('PADDLE_TRAINER_ID', 0))
    FLAGS.world_size = int(env.get('PADDLE_TRAINERS_NUM', 1))
    FLAGS.device_id = int(env['FLAGS_selected_gpus'])
    FLAGS.whole_batch_size = FLAGS.world_size * FLAGS.batch_size

    pipe = create_dali_pipeline(batch_size=FLAGS.batch_size,
                                num_threads=FLAGS.num_threads,
                                device_id=FLAGS.device_id,
                                data_dir=os.path.join(FLAGS.data, 'train'),
                                crop=224,
                                size=256,
                                dali_cpu=False,
                                shard_id=FLAGS.local_rank,
                                num_shards=FLAGS.world_size,
                                is_training=True)
    pipe.build()
    sample_per_shard = pipe.epoch_size("Reader") // FLAGS.world_size
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader")

    if FLAGS.local_rank == 0:
        pipe = create_dali_pipeline(batch_size=FLAGS.batch_size,
                                    num_threads=FLAGS.num_threads,
                                    device_id=FLAGS.device_id,
                                    data_dir=os.path.join(FLAGS.data, 'val'),
                                    crop=224,
                                    size=256,
                                    dali_cpu=False,
                                    shard_id=0,
                                    num_shards=1,
                                    is_training=False)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, reader_name="Reader")

    place = fluid.CUDAPlace(FLAGS.device_id)
    exe = fluid.Executor(place)
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    eval_prog = fluid.Program()

    step_per_epoch = int(math.ceil(sample_per_shard / FLAGS.batch_size))
    milestones = [step_per_epoch * e for e in (30, 60, 80)]
    values = [FLAGS.lr * (0.1**i) for i in range(len(milestones) + 1)]

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_fetch_list = build()
            learning_rate = fluid.layers.piecewise_decay(
                boundaries=milestones, values=values)
            learning_rate = fluid.layers.linear_lr_warmup(
                learning_rate=learning_rate,
                warmup_steps=5 * step_per_epoch,
                start_lr=0.,
                end_lr=FLAGS.lr)
            decay = FLAGS.weight_decay
            optimizer = fluid.optimizer.Momentum(
                learning_rate=learning_rate,
                momentum=FLAGS.momentum,
                regularization=fluid.regularizer.L2Decay(decay))
            avg_loss = train_fetch_list[0]
            optimizer.minimize(avg_loss)

    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_fetch_list = build()
        eval_prog = eval_prog.clone(True)

    build_strategy = fluid.BuildStrategy()
    build_strategy.trainer_id = FLAGS.local_rank
    build_strategy.num_trainers = FLAGS.world_size
    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        FLAGS.local_rank,
        trainers=os.environ.get('PADDLE_TRAINER_ENDPOINTS'),
        current_endpoint=os.environ.get('PADDLE_CURRENT_ENDPOINT'),
        startup_program=startup_prog,
        program=train_prog)

    exec_strategy = fluid.ExecutionStrategy()

    exe.run(startup_prog)
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)

    total_time = AverageMeter()

    for epoch in range(FLAGS.epochs):
        if FLAGS.local_rank == 0:
            print("==== train epoch {:02d} ====".format(epoch + 1))
        avg_time, _, _ = run(
            exe, compiled_train_prog, train_fetch_list, train_loader, epoch)
        total_time.update(avg_time)
        # reset DALI iterators
        train_loader.reset()

        if FLAGS.local_rank == 0:
            print("==== validation epoch {:02d} ====".format(epoch + 1))
            _, prec1, prec5 = run(
                exe, compiled_eval_prog, eval_fetch_list, val_loader, epoch)

            val_loader.reset()

            ckpt_path = os.path.join('checkpoint', "{:02d}".format(epoch + 1))
            if os.path.isdir(ckpt_path):
                shutil.rmtree(ckpt_path)
            print('Save model to {}.'.format(ckpt_path))
            fluid.io.save_persistables(exe, ckpt_path, train_prog)

            time_per_sample = FLAGS.whole_batch_size / total_time.avg
            if epoch == FLAGS.epochs-1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(
                          prec1 * 100, prec5 * 100, time_per_sample))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument('-d', '--depth', default=50, type=int,
                        metavar='N', help='number of layers (default: 50)')
    parser.add_argument('-j', '--num_threads', default=4, type=int,
                        metavar='N', help='number of threads (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--epochs', default=90, type=int,
                        metavar='N', help='number of epochs to be run (default 90)')
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    # In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode.
    # So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.
    paddle.enable_static()
    main()
