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

class DecoderPipeline(Pipeline):
    def __init__(self, data_path, batch_size, num_threads, device_id, decoder):
        super(DecoderPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=1)
        self.input = ops.FileReader(file_root = data_path,
                                    shard_id = 0,
                                    num_shards = 1)
        if decoder == 'nvJPEGDecoder':
            self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        elif decoder == 'HostDecoder':
            self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        output = self.decode(inputs)
        return (output, labels)

test_data_root = os.environ['DALI_EXTRA_PATH']
good_path = 'db/single'
missnamed_path = 'db/single/missnamed'
test_good_path = {'jpeg', 'mixed', 'png', 'tiff', 'pnm', 'bmp'}
test_missnamed_path = {'jpeg', 'png', 'tiff', 'pnm', 'bmp'}

def run_decode(data_path, batch, decoder, threads):
    pipe = DecoderPipeline(data_path=data_path, batch_size=batch, num_threads=threads, device_id=0, decoder=decoder)
    pipe.build()
    iters = pipe.epoch_size("Reader")
    for _ in range(iters):
        pipe.run()

def test_host_decoder():
    for decoder in {'nvJPEGDecoder', 'HostDecoder'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_good_path:
                    data_path = os.path.join(test_data_root, good_path, img_type)
                    run_decode(data_path, size, decoder, threads)
                    yield check, img_type, size, decoder, threads

def test_missnamed_host_decoder():
    for decoder in {'nvJPEGDecoder', 'HostDecoder'}:
        for threads in {1, 2, 3, 4}:
            for size in {1, 10}:
                for img_type in test_missnamed_path:
                    data_path = os.path.join(test_data_root, missnamed_path, img_type)
                    run_decode(data_path, size, decoder, threads)
                    yield check, img_type, size, decoder, threads

def check(img_type, size, decoder, threads):
    pass

