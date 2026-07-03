# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def
from nvidia.dali import fn


@pipeline_def
def training_pipeline(image_dir):
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True)
    images = fn.decoders.image(jpegs, device="mixed")
    angle = fn.random.uniform(range=(-30, 30))
    images = fn.rotate(images, angle=angle)
    mirror = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(
        images,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    return images, labels


pipe = training_pipeline(
    image_dir="/data/images",
    batch_size=64,
    num_threads=4,
    device_id=0,
    seed=42,
)
pipe.build()
for _ in range(100):
    images, labels = pipe.run()
