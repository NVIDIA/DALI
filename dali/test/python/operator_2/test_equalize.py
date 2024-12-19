# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import math
import itertools

import cv2
import numpy as np

from nvidia.dali import pipeline_def, fn, types
from test_utils import get_dali_extra_path, check_batch
from nose2.tools import params

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")


def equalize_cv_baseline(img, layout):
    if layout == "HW":
        return cv2.equalizeHist(img)
    if layout == "HWC":
        img = img.transpose(2, 0, 1)
        axis = 2
    else:
        assert layout == "CHW", f"{layout}"
        axis = 0
    return np.stack([cv2.equalizeHist(channel) for channel in img], axis=axis)


@pipeline_def
def images_pipeline(layout, dev):
    images, _ = fn.readers.file(
        name="Reader", file_root=images_dir, prefetch_queue_depth=2, random_shuffle=True, seed=42
    )
    decoder = "mixed" if dev == "gpu" else "cpu"
    if layout == "HW":
        images = fn.decoders.image(images, device=decoder, output_type=types.GRAY)
        images = fn.squeeze(images, axes=2)
    else:
        assert layout in ["HWC", "CHW"], f"{layout}"
        images = fn.decoders.image(images, device=decoder, output_type=types.RGB)
        if layout == "CHW":
            images = fn.transpose(images, perm=[2, 0, 1])
    equalized = fn.experimental.equalize(images)
    return equalized, images


@params(
    *tuple(
        itertools.product(
            ("cpu", "gpu"),
            (("HWC", 1), ("HWC", 32), ("CHW", 1), ("CHW", 7), ("HW", 253), ("HW", 128)),
        )
    )
)
def test_image_pipeline(dev, layout_batch_size):
    layout, batch_size = layout_batch_size
    num_iters = 2

    pipe = images_pipeline(
        num_threads=4, device_id=0, batch_size=batch_size, layout=layout, dev=dev
    )

    for _ in range(num_iters):
        equalized, imgs = pipe.run()
        if dev == "gpu":
            imgs = imgs.as_cpu()
            equalized = equalized.as_cpu()
        equalized = [np.array(img) for img in equalized]
        imgs = [np.array(img) for img in imgs]
        assert len(equalized) == len(imgs)
        baseline = [equalize_cv_baseline(img, layout) for img in imgs]
        check_batch(equalized, baseline, max_allowed_error=1)


@params(("cpu",), ("gpu",))
def test_multichannel(dev):
    sizes = [(200, 300), (700, 500), (1024, 200), (200, 1024), (1024, 1024)]
    num_channels = [1, 2, 3, 4, 5, 13]
    # keep len(sizes) and len(num_channels) co-prime to have all combinations
    assert math.gcd(len(sizes), len(num_channels)) == 1
    batch_size = len(sizes) * len(num_channels)
    rng = np.random.default_rng(424242)
    num_iters = 2

    def input_sample(sample_info):
        idx_in_batch = sample_info.idx_in_batch
        size = sizes[idx_in_batch % len(sizes)]
        num_channel = num_channels[idx_in_batch % len(num_channels)]
        shape = (size[0], size[1], num_channel)
        return np.uint8(rng.uniform(0, 255, shape))

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, seed=42)
    def pipeline():
        input = fn.external_source(input_sample, batch=False)
        if dev == "gpu":
            input = input.gpu()
        return fn.experimental.equalize(input), input

    pipe = pipeline()

    for _ in range(num_iters):
        equalized, imgs = pipe.run()
        if dev == "gpu":
            imgs = imgs.as_cpu()
            equalized = equalized.as_cpu()
        equalized = [np.array(img) for img in equalized]
        imgs = [np.array(img) for img in imgs]
        assert len(equalized) == len(imgs)
        baseline = [equalize_cv_baseline(img, "HWC") for img in imgs]
        check_batch(equalized, baseline, max_allowed_error=1)
