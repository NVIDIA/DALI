# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn
from nvidia.dali import types

from nvidia.dali.pipeline.experimental import pipeline_def

from nvidia.dali.auto_aug import auto_augment, trivial_augment


@pipeline_def(enable_conditionals=True)
def training_pipe(data_dir, interpolation, image_size, automatic_augmentation, dali_device="gpu",
                  rank=0, world_size=1):
    rng = fn.random.coin_flip(probability=0.5)

    jpegs, labels = fn.readers.file(name="Reader", file_root=data_dir, shard_id=rank,
                                    num_shards=world_size, random_shuffle=True, pad_last_batch=True)

    if dali_device == "gpu":
        decoder_device = "mixed"
        rrc_device = "gpu"
    else:
        decoder_device = "cpu"
        rrc_device = "cpu"

    images = fn.decoders.image(jpegs, device=decoder_device, output_type=types.RGB,
                               device_memory_padding=211025920, host_memory_padding=140544512)

    images = fn.random_resized_crop(images, device=rrc_device, size=[image_size, image_size],
                                    interp_type=interpolation,
                                    random_aspect_ratio=[0.75, 4.0 / 3.0], random_area=[0.08, 1.0],
                                    num_attempts=100, antialias=False)

    # Make sure that from this point we are processing on GPU regardless of dali_device parameter
    images = images.gpu()

    images = fn.flip(images, horizontal=rng)

    # Based on the specification, apply the automatic augmentation policy. Note, that from the point
    # of Pipeline definition, this `if` statement relies on static scalar parameter, so it is
    # evaluated exactly once during build - we either include automatic augmentations or not.
    if automatic_augmentation == "autoaugment":
        shapes = fn.peek_image_shape(jpegs)
        output = auto_augment.auto_augment_image_net(images, shapes)
    elif automatic_augmentation == "trivialaugment":
        output = trivial_augment.trivial_augment_wide(images)
    else:
        output = images

    output = fn.crop_mirror_normalize(output, dtype=types.FLOAT, output_layout=types.NCHW,
                                      crop=(image_size, image_size),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    return output, labels


@pipeline_def
def validation_pipe(data_dir, interpolation, image_size, image_crop, rank=0, world_size=1):
    jpegs, label = fn.readers.file(name="Reader", file_root=data_dir, shard_id=rank,
                                   num_shards=world_size, random_shuffle=False, pad_last_batch=True)

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_shorter=image_size, interp_type=interpolation,
                       antialias=False)

    output = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout=types.NCHW,
                                      crop=(image_crop, image_crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return output, label
