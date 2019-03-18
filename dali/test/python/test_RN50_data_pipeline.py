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
import nvidia.dali.tfrecord as tfrec
import glob
import argparse
import time

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, prefetch, fp16, nhwc):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=prefetch)

        self.decode_gpu = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size =(224,224))
        if nhwc:
            layout = types.NHWC
        else:
            layout = types.NCHW
        if fp16:
            out_type = types.FLOAT16
        else:
            out_type = types.FLOAT
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=out_type,
                                            output_layout=layout,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def base_define_graph(self, inputs, labels):
        rng = self.coin()
        images = self.decode_gpu(inputs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return (output, labels)

class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, fp16, nhwc)
        self.input = ops.MXNetReader(path = data_paths[0], index_path=data_paths[1], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class CaffeReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(CaffeReadPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, fp16, nhwc)
        self.input = ops.CaffeReader(path = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class Caffe2ReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(Caffe2ReadPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, fp16, nhwc)
        self.input = ops.Caffe2Reader(path = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class FileReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, fp16, nhwc)
        self.input = ops.FileReader(file_root = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class TFRecordPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, fp16, nhwc)
        tfrecord = sorted(glob.glob(data_paths[0]))
        tfrecord_idx = sorted(glob.glob(data_paths[1]))
        self.input = ops.TFRecordReader(path = tfrecord,
                                        index_path = tfrecord_idx,
                                        shard_id = device_id,
                                        num_shards = num_gpus,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
                                        })

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        return self.base_define_graph(images, labels)

test_data = {
#            FileReadPipeline: [["/data/imagenet/train-jpeg"],
#                               ["/data/imagenet/val-jpeg"]],
            MXNetReaderPipeline: [["/data/imagenet/train-480-val-256-recordio/train.rec", "/data/imagenet/train-480-val-256-recordio/train.idx"],
                                   ["/data/imagenet/train-480-val-256-recordio/val.rec", "/data/imagenet/train-480-val-256-recordio/val.idx"]],
#            CaffeReadPipeline: [["/data/imagenet/train-lmdb-256x256"],
#                                 ["/data/imagenet/val-lmdb-256x256"]],
#            Caffe2ReadPipeline: [["/data/imagenet/train-c2lmdb-480"],
#                                  ["/data/imagenet/val-c2lmdb-256"]],
#            TFRecordPipeline: [["/data/imagenet/train-val-tfrecord-480/train-*", "/data/imagenet/train-val-tfrecord-480.idx/train-*"]],
            }

parser = argparse.ArgumentParser(description='Test nvJPEG based RN50 augmentation pipeline with different datasets')
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                    help='number of GPUs (default: 1)')
parser.add_argument('-b', '--batch', default=2048, type=int, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 3)')
parser.add_argument('--prefetch', default=2, type=int, metavar='N',
                    help='prefetch queue deptch (default: 2)')
parser.add_argument('--fp16', action='store_true',
                    help='Run fp16 pipeline')
parser.add_argument('--nhwc', action='store_true',
                    help='Use NHWC data instead of default NCHW')
parser.add_argument('-i', '--iters', default=-1, type=int, metavar='N',
                    help='Number of iterations to run (default: -1 - whole data set)')
parser.add_argument('-cpu', '--cpu_size', type=int, metavar='N',
                    help='cpu queue size, must be used with -gpu to force separte execution')
parser.add_argument('-gpu', '--gpu_size', type=int, metavar='N',
                    help='gpu queue size, must be used with -cpu to force separte execution')

args = parser.parse_args()

N = args.gpus             # number of GPUs
BATCH_SIZE = args.batch   # batch size
LOG_INTERVAL = args.print_freq
if args.cpu_size is not None and args.gpu_size is not None:
    PREFETCH = {"cpu_size": args.cpu_size, "gpu_size": args.gpu_size}
else:
    PREFETCH = args.prefetch
WORKERS = args.workers
FP16 = args.fp16
NHWC = args.nhwc

print("GPUs: {}, batch: {}, workers: {}, prefetch depth: {}, loging interval: {}, fp16: {}, NHWC: {}"
      .format(N, BATCH_SIZE, WORKERS, PREFETCH, LOG_INTERVAL, FP16, NHWC))

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

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=n,
                           num_gpus=N, data_paths=data_set, prefetch=PREFETCH, fp16=FP16, nhwc=NHWC) for n in range(N)]
        [pipe.build() for pipe in pipes]

        if args.iters < 0:
            iters = pipes[0].epoch_size("Reader")
            assert(all(pipe.epoch_size("Reader") == iters for pipe in pipes))
            iters_tmp = iters
            iters = iters // BATCH_SIZE
            if iters_tmp != iters * BATCH_SIZE:
                iters += 1
            iters_tmp = iters

            iters = iters // N
            if iters_tmp != iters * N:
                iters += 1
        else:
            iters = args.iters

        print ("RUN {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
        print (data_set)
        end = time.time()
        for i in range(2):
          if i == 0:
              print("Warm up")
          elif i == 1:
              print("Test run")
          data_time = AverageMeter()
          for j in range(iters):
              for pipe in pipes:
                  pipe.run()
              data_time.update(time.time() - end)
              if j % LOG_INTERVAL == 0:
                  print("{} {}/ {}, avg time: {} [s], worst time: {} [s], speed: {} [img/s]"
                  .format(pipe_name.__name__, j + 1, iters, data_time.avg, data_time.max_val, N * BATCH_SIZE / data_time.avg))
              end = time.time()

        print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))

