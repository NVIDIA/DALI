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
import argparse
import numpy as np
import random

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, prefetch, seed):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=prefetch)

        self.decode_cpu = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.decode_crop = ops.HostDecoderSlice(device = "cpu", output_type = types.RGB)
        self.crop = ops.SSDRandomCrop(device="cpu", num_attempts=1, seed=seed)
        self.crop2 = ops.RandomBBoxCrop(device="cpu",
                                        aspect_ratio=[0.5, 2.0],
                                        thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                        scaling=[0.3, 1.0],
                                        ltrb=True,
                                        seed=seed)
        self.slice_cpu = ops.Slice(device="cpu")
        self.slice_gpu = ops.Slice(device="gpu")

    def base_define_graph(self, inputs, labels, bboxes):
        images = self.decode_cpu(inputs)
        images_1, bboxes_1, labels_1 = self.crop(images, bboxes, labels)
        crop_begin, crop_size, bboxes_2, labels_2 = self.crop2(bboxes, labels)
        images_2 = self.decode_crop(inputs, crop_begin, crop_size)
        images_3 = self.slice_cpu(images, crop_begin, crop_size)
        images_4 = self.slice_gpu(images.gpu(), crop_begin, crop_size)
        return (bboxes_1, labels_1, images_1, bboxes_2, labels_2, images_2, images_3, images_4)

class COCOReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, seed):
        super(COCOReaderPipeline, self).__init__(batch_size, num_threads, device_id, prefetch, seed)
        self.input = ops.COCOReader(file_root = data_paths[0], annotations_file=data_paths[1],
                                    shard_id = device_id, num_shards = num_gpus, ratio=True, ltrb=True,
                                    random_shuffle=True)

    def define_graph(self):
        images, bb, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels, bb)

test_data = {
  COCOReaderPipeline: [["/data/coco/coco-2017/coco2017/train2017", "/data/coco/coco-2017/coco2017/annotations/instances_train2017.json"],
                       ["/data/coco/coco-2017/coco2017/val2017", "/data/coco/coco-2017/coco2017/annotations/instances_val2017.json"]]
            }

parser = argparse.ArgumentParser(description='Random crop for bounding boxes test')
parser.add_argument('-i', '--iters', default=-1, type=int, metavar='N',
                    help='Number of iterations to run (default: -1 - whole data set)')
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N',
                    help='number of GPUs (default: 1)')
args = parser.parse_args()

N = args.gpus             # number of GPUs
BATCH_SIZE = 1
LOG_INTERVAL = 100
WORKERS = 3
PREFETCH = 2

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        seed = int(random.random() * (1<<64))
        print ("Seed:", seed)
        print("Build pipeline")
        pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=WORKERS, device_id=n,
                           num_gpus=N, data_paths=data_set, prefetch=PREFETCH, seed=1) for n in range(N)]
        [pipe.build() for pipe in pipes]
        if args.iters < 0:
          iters = pipes[0].epoch_size("Reader")
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
        for j in range(iters):
            for pipe in pipes:
                bboxes_1, labels_1, images_1, bboxes_2, labels_2, images_2, images_3, images_4 = pipe.run()
                bboxes_1_arr = np.squeeze(bboxes_1.as_array())
                bboxes_2_arr = np.squeeze(bboxes_2.as_array())
                labels_1_arr = np.squeeze(labels_1.as_array())
                labels_2_arr = np.squeeze(labels_2.as_array())
                images_1_arr = images_1.as_array()
                images_2_arr = images_2.as_array()
                images_3_arr = images_3.as_array()
                images_4_arr = images_4.asCPU().as_array()
                res = np.allclose(labels_1_arr, labels_2_arr)
                if not res:
                    print(labels_1_arr, "\nvs\n", labels_2_arr)
                res_bb = np.allclose(bboxes_1_arr, bboxes_2_arr)
                if not res_bb:
                    print(bboxes_1_arr, "\nvs\n", bboxes_2_arr)
                res_img = np.allclose(images_1_arr, images_2_arr) and np.allclose(images_1_arr, images_3_arr) and np.allclose(images_1_arr, images_4_arr)
                if not res_img:
                    print(images_1_arr, "\nvs\n", images_2_arr)
                    print(images_1_arr, "\nvs\n", images_3_arr)
                    print(images_1_arr, "\nvs\n", images_4_arr)
                if not res_bb or not res or not res_img:
                    print("Labels == ", res)
                    print("Bboxes == ", res_bb)
                    print("Images == ", res_img)
                    exit(1)
            if not j % LOG_INTERVAL:
                print("{} {}/ {}".format(pipe_name.__name__, j + 1, iters))

        print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))

