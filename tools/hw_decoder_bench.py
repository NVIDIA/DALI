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

import argparse
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import time
from nvidia.dali.pipeline import pipeline_def
import random
import numpy as np
import os
from nvidia.dali.auto_aug import auto_augment

parser = argparse.ArgumentParser(description='DALI HW decoder benchmark')
parser.add_argument('-b', dest='batch_size', help='batch size', default=1, type=int)
parser.add_argument('-d', dest='device_id', help='device id', default=0, type=int)
parser.add_argument('-n', dest='gpu_num',
                    help='Number of GPUs used starting from device_id', default=1, type=int)
parser.add_argument('-g', dest='device', choices=['gpu', 'cpu'],
                    help='device to use', default='gpu',
                    type=str)
parser.add_argument('-w', dest='warmup_iterations', help='warmup iterations', default=0, type=int)
parser.add_argument('-t', dest='total_images', help='total images', default=100, type=int)
parser.add_argument('-j', dest='num_threads', help='num_threads', default=1, type=int)
input_files_arg = parser.add_mutually_exclusive_group()
input_files_arg.add_argument('-i', dest='images_dir', help='images dir')
input_files_arg.add_argument('--image_list', dest='image_list', nargs='+', default=[],
                             help='List of images used for the benchmark.')
parser.add_argument('-p', dest='pipeline', choices=['decoder', 'rn50', 'efficientnet_inference',
                                                    'vit'],
                    help='pipeline to test', default='decoder',
                    type=str)
parser.add_argument('--width_hint', dest='width_hint', default=0, type=int)
parser.add_argument('--height_hint', dest='height_hint', default=0, type=int)
parser.add_argument('--hw_load', dest='hw_load',
                    help='HW decoder workload (e.g. 0.66 means 66% of the batch)', default=0.75,
                    type=float)
args = parser.parse_args()

DALI_INPUT_NAME = 'DALI_INPUT_0'
needs_feed_input = args.pipeline == 'efficientnet_inference'


@pipeline_def(batch_size=args.batch_size,
              num_threads=args.num_threads,
              device_id=args.device_id,
              seed=0)
def DecoderPipeline():
    device = 'mixed' if args.device == 'gpu' else 'cpu'
    jpegs, _ = fn.readers.file(file_root=args.images_dir)
    images = fn.decoders.image(jpegs, device=device, output_type=types.RGB,
                               hw_decoder_load=args.hw_load, preallocate_width_hint=args.width_hint,
                               preallocate_height_hint=args.height_hint)
    return images


@pipeline_def(batch_size=args.batch_size,
              num_threads=args.num_threads,
              device_id=args.device_id,
              seed=0)
def RN50Pipeline():
    device = 'mixed' if args.device == 'gpu' else 'cpu'
    jpegs, _ = fn.readers.file(file_root=args.images_dir)
    images = fn.decoders.image_random_crop(jpegs, device=device, output_type=types.RGB,
                                           hw_decoder_load=args.hw_load,
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
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=coin_flip)
    return images


@pipeline_def(batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id,
              prefetch_queue_depth=1)
def EfficientnetInferencePipeline():
    images = fn.external_source(device='cpu', name=DALI_INPUT_NAME)
    images = fn.decoders.image(images, device='mixed' if args.device == 'gpu' else 'cpu',
                               output_type=types.RGB, hw_decoder_load=args.hw_load)
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      output_layout='CHW',
                                      crop=(224, 224),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return images


def feed_input(dali_pipeline, data):
    if needs_feed_input:
        assert data is not None, "Input data has not been provided."
        dali_pipeline.feed_input(DALI_INPUT_NAME, data)


def create_input_tensor(batch_size, file_list):
    """
    Creates an input batch to the DALI Pipeline.
    The batch will comprise the files defined within file list and will be shuffled.
    If the file list contains fewer files than the batch size, they will be repeated.
    The encoded images will be padded.
    :param batch_size: Requested batch size.
    :param file_list: List of images to be loaded.
    :return:
    """
    # Adjust file_list to batch_size
    while len(file_list) < batch_size:
        file_list += file_list
    file_list = file_list[:batch_size]

    random.shuffle(file_list)

    # Read the files as byte buffers
    arrays = list(map(lambda x: np.fromfile(x, dtype=np.uint8), file_list))

    # Pad the encoded images
    lengths = list(map(lambda x, ar=arrays: ar[x].shape[0], [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(map(lambda ar, ml=max_len: np.pad(ar, (0, ml - ar.shape[0])), arrays))

    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def non_image_preprocessing(raw_text):
    return np.array([int(bytes(raw_text).decode('utf-8'))])


@pipeline_def(batch_size=args.batch_size,
              num_threads=args.num_threads,
              device_id=args.device_id,
              seed=0)
def vit_pipeline(is_training=False, image_shape=(384, 384, 3), num_classes=1000):
    files_paths = [os.path.join(args.images_dir, f) for f in os.listdir(
        args.images_dir)]

    img, clss = fn.readers.webdataset(
        paths=files_paths,
        index_paths=None,
        ext=['jpg', 'cls'],
        missing_component_behavior='error',
        random_shuffle=False,
        shard_id=0,
        num_shards=1,
        pad_last_batch=False if is_training else True,
        name='webdataset_reader')

    use_gpu = args.device == 'gpu'
    labels = fn.python_function(clss, function=non_image_preprocessing, num_outputs=1)
    if use_gpu:
        labels = labels.gpu()
    labels = fn.one_hot(labels, num_classes=num_classes)

    device = 'mixed' if use_gpu else 'cpu'
    img = fn.decoders.image(img, device=device, output_type=types.RGB,
                            hw_decoder_load=args.hw_load,
                            preallocate_width_hint=args.width_hint,
                            preallocate_height_hint=args.height_hint)

    if is_training:
        img = fn.random_resized_crop(img, size=image_shape[:-1])
        img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip())

        # color jitter
        brightness = fn.random.uniform(range=[0.6, 1.4])
        contrast = fn.random.uniform(range=[0.6, 1.4])
        saturation = fn.random.uniform(range=[0.6, 1.4])
        hue = fn.random.uniform(range=[0.9, 1.1])
        img = fn.color_twist(
            img,
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation)

        # auto-augment
        # `shape` controls the magnitude of the translation operations
        img = auto_augment.auto_augment_image_net(img)
    else:
        img = fn.resize(img, size=image_shape[:-1])

    # normalize
    # https://github.com/NVIDIA/DALI/issues/4469
    mean = np.asarray([0.485, 0.456, 0.406])[None, None, :]
    std = np.asarray([0.229, 0.224, 0.225])[None, None, :]
    scale = 1 / 255.
    img = fn.normalize(
        img,
        mean=mean / scale,
        stddev=std,
        scale=scale,
        dtype=types.FLOAT)

    return img, labels


pipes = []
if args.pipeline == 'decoder':
    for i in range(args.gpu_num):
        pipes.append(DecoderPipeline(device_id=i + args.device_id))
elif args.pipeline == 'rn50':
    for i in range(args.gpu_num):
        pipes.append(RN50Pipeline(device_id=i + args.device_id))
elif args.pipeline == 'efficientnet_inference':
    for i in range(args.gpu_num):
        pipes.append(EfficientnetInferencePipeline(device_id=i + args.device_id))
elif args.pipeline == 'vit':
    for i in range(args.gpu_num):
        pipes.append(vit_pipeline(device_id=i + args.device_id))
else:
    raise RuntimeError('Unsupported pipeline')
for p in pipes:
    p.build()

input_tensor = create_input_tensor(args.batch_size, args.image_list) if needs_feed_input else None

for iteration in range(args.warmup_iterations):
    for p in pipes:
        feed_input(p, input_tensor)
        p.schedule_run()
    for p in pipes:
        _ = p.share_outputs()
    for p in pipes:
        p.release_outputs()
print('Warmup finished')

start = time.time()
test_iterations = args.total_images // args.batch_size

print('Test iterations: ', test_iterations)
for iteration in range(test_iterations):
    for p in pipes:
        feed_input(p, input_tensor)
        p.schedule_run()
    for p in pipes:
        _ = p.share_outputs()
    for p in pipes:
        p.release_outputs()
end = time.time()
total_time = end - start

print(test_iterations * args.batch_size * args.gpu_num / total_time, 'fps')
