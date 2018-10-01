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

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)
        self.decode_host = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.prospective_crop = ops.BBoxCrop(device='cpu',
                                             thresholds=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                                             aspect_ratio=[0.5, 2.0],
                                             scaling=[0.3, 1.0])
        self.crop = ops.Crop(device='cpu')

    def base_define_graph(self, inputs, labels, bboxes):
        images_host = self.decode_host(inputs)
        crop_begin, crop_size, bb = self.prospective_crop(images_host, bboxes)
        images_host = self.crop(images_host, crop_begin, crop_size)

        return (images_host, bb)

class COCOReaderPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, data_paths):
        super(COCOReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.COCOReader(file_root = data_paths[0], annotations_file=data_paths[1])

    def define_graph(self):
        images, bb, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels, bb)

test_data = {
            COCOReaderPipeline: [["/data/coco/coco-2017/coco2017/train2017", "/data/coco/coco-2017/coco2017/annotations/instances_train2017.json", 118288],
                                ["/data/coco/coco-2017/coco2017/val2017", "/data/coco/coco-2017/coco2017/annotations/instances_val2017.json", 5001]]
            }

N = 1               # number of GPUs
BATCH_SIZE = 1024   # batch size
LOG_INTERVAL = 200 // BATCH_SIZE + 1

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        pipes = [pipe_name(batch_size=BATCH_SIZE, num_threads=4, device_id = n, num_gpus = N, data_paths = data_set) for n in range(N)]
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
                pipe._start_run()
            for pipe in pipes:
                pipe.outputs()
            if j % LOG_INTERVAL == 0:
                print (pipe_name.__name__, j + 1, "/", iters)

        print("OK {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
