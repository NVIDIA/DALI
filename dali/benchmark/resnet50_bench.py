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

import argparse
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from timeit import default_timer as timer

image_folder = "/data/dali/benchmark/benchmark_images"

def read_jpegs(folder):
    with open(folder + "/image_list.txt", 'r') as file:
        files = [line.rstrip() for line in file]

    images = []
    for fname in files:
        f = open(image_folder + "/" + fname, 'rb')
        images.append(np.fromstring(f.read(), dtype = np.uint8))
    return images

def make_batch(size):
    data = read_jpegs(image_folder)
    return [data[i % len(data)] for i in range(size)]

class C2Pipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pipelined = True, exec_async = True):
        super(C2Pipe, self).__init__(batch_size,
                                     num_threads,
                                     device_id,
                                     exec_pipelined=pipelined,
                                     exec_async=exec_async)
        self.input = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)
        self.rcm = ops.FastResizeCropMirror(crop = (224, 224))
        self.np = ops.CropMirrorNormalize(device = "gpu",
                                          dtype = types.FLOAT16,
                                          mean = [128., 128., 128.],
                                          std = [1., 1., 1.])
        self.uniform = ops.random.Uniform(range = (0., 1.))
        self.resize_uniform = ops.random.Uniform(range = (256., 480.))
        self.mirror = ops.random.CoinFlip(probability = 0.5)

    def define_graph(self):
        self.jpegs = self.input()
        images = self.decode(self.jpegs)
        resized = self.rcm(images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform(),
                           mirror = self.mirror(),
                           resize_shorter = self.resize_uniform())
        output = self.np(resized.gpu())
        return output

    def iter_setup(self):
        raw_data = make_batch(self.batch_size)
        self.feed_input(self.jpegs, raw_data)

class HybridPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pipelined = True, exec_async = True):
        super(HybridPipe, self).__init__(batch_size,
                                         num_threads,
                                         device_id,
                                         exec_pipelined=pipelined,
                                         exec_async=exec_async)
        self.input = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu",
                                 interp_type = types.INTERP_LINEAR)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            dtype = types.FLOAT16,
                                            crop = (224, 224),
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.uniform = ops.random.Uniform(range = (0., 1.))
        self.resize_uniform = ops.random.Uniform(range = (256., 480.))
        self.mirror = ops.random.CoinFlip(probability = 0.5)

    def define_graph(self):
        self.jpegs = self.input()
        images = self.decode(self.jpegs)
        resized = self.resize(images, resize_shorter = self.resize_uniform())
        output = self.cmnp(resized, mirror = self.mirror(),
                           crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform())
        return output

    def iter_setup(self):
        raw_data = make_batch(self.batch_size)
        self.feed_input(self.jpegs, raw_data)

def run_benchmarks(PipeType, args):
    print("Running Benchmarks For {}".format(PipeType.__name__))
    for executor in args.executors:
        pipelined = executor > 0
        exec_async = executor > 1
        for batch_size in args.batch_sizes:
            for num_threads in args.thread_counts:
                pipe = PipeType(batch_size, num_threads, 0, pipelined, exec_async)
                pipe.build()
                start_time = timer()
                for i in range(args.num_iters):
                    pipe.run()

                total_time = timer() - start_time
                print("{}/{}/{}/{}: FPS={}"
                      .format(PipeType.__name__,  executor, batch_size, num_threads,
                              float(batch_size * args.num_iters) / total_time))

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-sizes', default = [128],
                        help='Comma separated list of batch sizes to run')
    parser.add_argument('--thread-counts', default = [1,2,3,4],
                        help='Comma separated list of thread counts')
    parser.add_argument('--executors', default = [2],
                        help='List of executors to run')
    parser.add_argument('--num-iters', type=int, default=100,
                        help='Number of iterations to run')
    return parser.parse_args()

def main():
    args = get_args()
    pipe_types = [C2Pipe, HybridPipe]
    for PipeType in pipe_types:
        run_benchmarks(PipeType, args)

if __name__ == '__main__':
    main()
