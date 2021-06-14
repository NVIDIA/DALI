# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import argparse
import time

parser = argparse.ArgumentParser(description='DALI HW decoder benchmark')
parser.add_argument('-b', dest='batch_size', help='batch size', default=1, type=int)
parser.add_argument('-d', dest='device_id', help='device id', default=0, type=int)
parser.add_argument('-g', dest='device', help='device to use', choices=['gpu', 'cpu'], default='gpu', type=str)
parser.add_argument('-w', dest='warmup_iterations', help='warmup iterations', default=0, type=int)
parser.add_argument('-t', dest='total_images', help='total images', default=100, type=int)
parser.add_argument('-j', dest='num_threads', help='num_threads', default=1, type=int)
parser.add_argument('-i', dest='images_dir', help='images dir')
parser.add_argument('-p', dest='pipeline', help='pipeline to test', choices=['decoder', 'rn50'], default='decoder', type=str)
parser.add_argument('--width_hint', dest="width_hint", default=0, type=int)
parser.add_argument('--height_hint', dest="height_hint", default=0, type=int)
parser.add_argument('--hw_load', dest='hw_load', help='HW decoder workload (e.g. 0.66 means 66% of the batch)', default=0.75, type=float)
args = parser.parse_args()


@pipeline_def(batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id, seed=0)
def DecoderPipeline():
    device =  'mixed' if args.device == 'gpu' else 'cpu'
    jpegs, _ = fn.readers.file(file_root = args.images_dir)
    images = fn.decoders.image(jpegs, device = device, output_type = types.RGB, hw_decoder_load=args.hw_load,
                        preallocate_width_hint=args.width_hint,
                        preallocate_height_hint=args.height_hint)
    return images

@pipeline_def(batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id, seed=0)
def RN50Pipeline():
    device =  'mixed' if args.device == 'gpu' else 'cpu'
    jpegs, _ = fn.readers.file(file_root = args.images_dir)
    images = fn.decoders.image_random_crop(jpegs, device = device, output_type = types.RGB, hw_decoder_load=args.hw_load,
                        preallocate_width_hint=args.width_hint,
                        preallocate_height_hint=args.height_hint)
    images = fn.resize(images, resize_x=224, resize_y=224)
    layout = types.NCHW
    out_type = types.FLOAT16
    coin_flip = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(
        images,
        dtype=out_type,
        output_layout=layout,
        crop=(224, 224),
        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
        std=[0.229 * 255,0.224 * 255,0.225 * 255],
        mirror=coin_flip)
    return images

if args.pipeline == 'decoder':
    pipe = DecoderPipeline()
elif args.pipeline == 'rn50':
    pipe = RN50Pipeline()
else:
    raise RuntimeError('Unsupported pipeline')
pipe.build()

for iteration in range(args.warmup_iterations):
    output = pipe.run()
print('Warmup finished')

start = time.time()
test_iterations = args.total_images // args.batch_size

print('Test iterations: ', test_iterations)
for iteration in range(test_iterations):
    output = pipe.run()
end = time.time()
total_time = end - start

print(test_iterations * args.batch_size / total_time, 'fps')
