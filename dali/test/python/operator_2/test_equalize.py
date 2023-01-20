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

import os

import cv2
import numpy as np

from nvidia.dali import pipeline_def, fn, types
from test_utils import get_dali_extra_path, check_batch
from nose2.tools import params

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


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
def images_pipeline(layout):
    images, _ = fn.readers.file(name="Reader", file_root=images_dir, prefetch_queue_depth=2,
                                random_shuffle=True, seed=42)
    if layout == "HW":
        images = fn.decoders.image(images, device="mixed", output_type=types.GRAY)
        images = fn.squeeze(images, axes=2)
    else:
        assert layout in ["HWC", "CHW"], f"{layout}"
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        if layout == "CHW":
            images = fn.transpose(images, perm=[2, 0, 1])
    equalized = fn.experimental.equalize(images)
    return equalized, images


@params(("HWC", 1), ("HWC", 32), ("CHW", 1), ("CHW", 7), ("HW", 253), ("HW", 128))
def test_image_pipeline(layout, batch_size):
    num_iters = 2

    pipe = images_pipeline(num_threads=4, device_id=0, batch_size=batch_size, layout=layout)
    pipe.build()

    for _ in range(num_iters):
        equalized, imgs = pipe.run()
        equalized = [np.array(img) for img in equalized.as_cpu()]
        imgs = [np.array(img) for img in imgs.as_cpu()]
        assert len(equalized) == len(imgs)
        baseline = [equalize_cv_baseline(img, layout) for img in imgs]
        check_batch(equalized, baseline)
