# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode_gpu = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.decode_host = ops.HostDecoder(device = "cpu", output_type = types.RGB)

    def base_define_graph(self, inputs, labels):
        images_gpu = self.decode_gpu(inputs)
        images_host = self.decode_host(inputs)
        return (images_gpu, images_host, labels)

class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(path = data_paths[0], index_path=data_paths[1],
                                     shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class CaffeReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(CaffeReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.CaffeReader(path = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class Caffe2ReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(Caffe2ReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.Caffe2Reader(path = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class FileReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root = data_paths[0], shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

class TFRecordPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
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

class COCOReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(COCOReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.COCOReader(file_root = data_paths[0], annotations_file=data_paths[1],
                                    shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, bb, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

test_data = {
            FileReadPipeline: [["/data/imagenet/train-jpeg"],
                               ["/data/imagenet/val-jpeg"]],
            MXNetReaderPipeline: [["/data/imagenet/train-480-val-256-recordio/train.rec", "/data/imagenet/train-480-val-256-recordio/train.idx"],
                                   ["/data/imagenet/train-480-val-256-recordio/val.rec", "/data/imagenet/train-480-val-256-recordio/val.idx"]],
            CaffeReadPipeline: [["/data/imagenet/train-lmdb-256x256"],
                                 ["/data/imagenet/val-lmdb-256x256"]],
            Caffe2ReadPipeline: [["/data/imagenet/train-c2lmdb-480"],
                                  ["/data/imagenet/val-c2lmdb-256"]],
            TFRecordPipeline: [["/data/imagenet/train-val-tfrecord-480/train-*", "/data/imagenet/train-val-tfrecord-480.idx/train-*"]],
            COCOReaderPipeline: [["/data/coco/coco-2017/coco2017/train2017", "/data/coco/coco-2017/coco2017/annotations/instances_train2017.json"],
                                ["/data/coco/coco-2017/coco2017/val2017", "/data/coco/coco-2017/coco2017/annotations/instances_val2017.json"]]
            }

parser = argparse.ArgumentParser(description='nvJPEG and HostDecoder RN50 dataset test')
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                    help='number of GPUs (default: 1)')
parser.add_argument('-b', '--batch', default=2048, type=int, metavar='N',
                    help='batch size (default: 2048)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

N = args.gpus             # number of GPUs
BATCH_SIZE = args.batch   # batch size
LOG_INTERVAL = args.print_freq

print("GPUs: {}, batch: {}, loging interval: {}".format(N, BATCH_SIZE, LOG_INTERVAL))

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=4, device_id=n, num_gpus=N, data_paths=data_set) for n in range(N)]
        [pipe.build() for pipe in pipes]

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

        print ("RUN {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
        print (data_set)
        for j in range(iters):
            for pipe in pipes:
                pipe._run()
            for pipe in pipes:
                pipe.outputs()
            if j % LOG_INTERVAL == 0:
                print (pipe_name.__name__, j + 1, "/", iters)

        print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
