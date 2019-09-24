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
import os

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path

class DecoderPipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_threads, device_id, device, use_fast_idct=False):
        super(DecoderPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=1)
        self.input = ops.FileReader(file_root = data_path,
                                    shard_id = 0,
                                    num_shards = 1)
        self.decode = ops.ImageDecoder(device = device, output_type = types.RGB, use_fast_idct=use_fast_idct)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        output = self.decode(inputs)
        return (output, labels)

test_data_root = get_dali_extra_path()
good_path = 'db/single'
missnamed_path = 'db/single/missnamed'
test_good_path = {'jpeg', 'mixed', 'png', 'tiff', 'pnm', 'bmp'}
test_missnamed_path = {'jpeg', 'png', 'tiff', 'pnm', 'bmp'}

def run_decode(data_path, batch, device, threads):
    pipe = DecoderPipeline(data_path=data_path, batch_size=batch, num_threads=threads, device_id=0, device=device)
    pipe.build()
    iters = pipe.epoch_size("Reader")
    for _ in range(iters):
        pipe.run()

def test_image_decoder():
    for device in {'cpu', 'mixed'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_good_path:
                    data_path = os.path.join(test_data_root, good_path, img_type)
                    run_decode(data_path, size, device, threads)
                    yield check, img_type, size, device, threads

def test_missnamed_host_decoder():
    for decoder in {'cpu', 'mixed'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_missnamed_path:
                    data_path = os.path.join(test_data_root, missnamed_path, img_type)
                    run_decode(data_path, size, decoder, threads)
                    yield check, img_type, size, decoder, threads

def check(img_type, size, device, threads):
    pass

class DecoderPipelineFastIDC(Pipeline):
    def __init__(self, data_path, batch_size, num_threads, use_fast_idct=False):
        super(DecoderPipelineFastIDC, self).__init__(batch_size, num_threads, 0, prefetch_queue_depth=1)
        self.input = ops.FileReader(file_root = data_path,
                                    shard_id = 0,
                                    num_shards = 1)
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB, use_fast_idct=use_fast_idct)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        output = self.decode(inputs)
        return (output, labels)

def check_FastDCT_body(batch_size, img_type, device):
    data_path = os.path.join(test_data_root, good_path, img_type)
    compare_pipelines(DecoderPipeline(data_path=data_path, batch_size=batch_size, num_threads=3,
                                      device_id=0, device=device, use_fast_idct=False),
                      DecoderPipeline(data_path=data_path, batch_size=batch_size, num_threads=3,
                                      device_id=0, device='cpu', use_fast_idct=True),
                      # average difference should be no bigger by off-by-3
                      batch_size=batch_size, N_iterations=3, eps=3)

def check_FastDCT():
    for device in {'cpu', 'mixed'}:
        for batch_size in {1, 8}:
            for img_type in test_good_path:
              yield check_FastDCT_body, batch_size, img_type, device
