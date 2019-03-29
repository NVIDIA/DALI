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
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import glob
import argparse
import time
import tensorflow as tf
from nvidia.dali.plugin.mxnet import DALIClassificationIterator as MXNetIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
from nvidia.dali.plugin.tf import DALIIterator as TensorFlowIterator

data_paths = ["/data/imagenet/train-jpeg"]

class RN50Pipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(RN50Pipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=prefetch)
        self.input = ops.FileReader(file_root = data_paths[0], shard_id = device_id, num_shards = num_gpus)
        self.decode_gpu = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size =(224,224))

        layout = types.args.nhwc if nhwc else types.NCHW
        out_type = types.FLOAT16 if fp16 else types.FLOAT

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=out_type,
                                            output_layout=layout,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode_gpu(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return (output, labels.gpu())

parser = argparse.ArgumentParser(description='Test RN50 augmentation pipeline with different FW iterators')
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                    help='number of GPUs (default: 1)')
parser.add_argument('-b', '--batch_size', default=13, type=int, metavar='N',
                    help='batch size (default: 13)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 3)')
parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                    help='prefetch queue deptch (default: 2)')
parser.add_argument('--fp16', action='store_true',
                    help='Run fp16 pipeline')
parser.add_argument('--nhwc', action='store_true',
                    help='Use args.nhwc data instead of default NCHW')
parser.add_argument('-i', '--iters', default=-1, type=int, metavar='N',
                    help='Number of iterations to run (default: -1 - whole data set)')
parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                    help='Number of epochs to run (default: 1)')
args = parser.parse_args()

print("GPUs: {}, batch: {}, workers: {}, prefetch depth: {}, loging interval: {}, fp16: {}, args.nhwc: {}"
      .format(args.gpus, args.batch_size, args.workers, args.prefetch, args.print_freq, args.fp16, args.nhwc))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_last_n = 0
        self.max_val = 0

    def update(self, val, n=1):
        self.val = val
        self.max_val = max(self.max_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

Iterators = {"mxnet.DALIClassificationIterator"   : MXNetIterator,
             "pytorch.DALIClassificationIterator" : PyTorchIterator,
             "tf.DALIIterator"                    : TensorFlowIterator}
for iterator_name in Iterators:
    IteratorClass = Iterators[iterator_name]
    print("Start testing {}".format(iterator_name))

    pipes = [RN50Pipeline(batch_size=args.batch_size, num_threads=args.workers, device_id=n,
                          num_gpus=args.gpus, data_paths=data_paths, prefetch=args.prefetch,
                          fp16=args.fp16, nhwc=args.nhwc) for n in range(args.gpus)]
    [pipe.build() for pipe in pipes]
    iters = args.iters
    if args.iters < 0:
        iters = pipes[0].epoch_size("Reader")
        assert(all(pipe.epoch_size("Reader") == iters for pipe in pipes))
        iters_tmp = iters
        iters = iters // args.batch_size
        if iters_tmp != iters * args.batch_size:
            iters += 1
        iters_tmp = iters

        iters = iters // args.gpus
        if iters_tmp != iters * args.gpus:
            iters += 1

    sess = None
    images = []
    labels = []
    if iterator_name == "tf.DALIIterator":
        daliop = IteratorClass()
        for dev in range(args.gpus):
            with tf.device('/gpu:%i' % dev):
                image, label = daliop(pipeline = pipes[dev],
                    shapes = [(args.batch_size, 3, 224, 224), ()],
                    dtypes = [tf.int32, tf.float32])
                images.append(image)
                labels.append(label)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=config)

    end = time.time()
    for i in range(args.epochs):
        if i == 0:
            print("Warm up")
        else:
            print("Test run " + str(i))
        data_time = AverageMeter()

        if iterator_name == "tf.DALIIterator":
            assert sess != None
            for j in range(iters):
                res = sess.run([images, labels])
                data_time.update(time.time() - end)
                if j % args.print_freq == 0:
                    print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]"
                        .format(iterator_name, j + 1, iters, data_time.avg, data_time.max_val, args.gpus * args.batch_size / data_time.avg))
                end = time.time()
        else:
            dali_train_iter = IteratorClass(pipes, pipes[0].epoch_size("Reader"))
            j = 0
            for it in iter(dali_train_iter):
                data_time.update(time.time() - end)
                if j % args.print_freq == 0:
                    print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]"
                        .format(iterator_name, j + 1, iters, data_time.avg, data_time.max_val, args.gpus * args.batch_size / data_time.avg))
                end = time.time()
                j = j + 1
                if j > iters:
                    break
