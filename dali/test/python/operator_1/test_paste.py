# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import cv2
import numpy as np
from nose2.tools import cartesian_params
from nose_utils import raises
from nvidia.dali import fn, pipeline_def
from test_utils import get_dali_extra_path

data_root = Path(get_dali_extra_path())
img_dir = data_root / "db" / "single" / "jpeg"
file_list = img_dir / "image_list.txt"


def opencv_paste(image, ratio, paste_x, paste_y, fill_value):
    """Reimplement of paste operation with OpenCV copyMakeBorder"""
    h, w = image.shape[:2]
    paste_h = int(h * ratio)
    paste_w = int(w * ratio)
    top_pad = int(paste_y * (paste_h - h))
    bottom_pad = paste_h - h - top_pad
    left_pad = int(paste_x * (paste_w - w))
    right_pad = paste_w - w - left_pad

    return cv2.copyMakeBorder(
        image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value,
    )


@cartesian_params(["gpu", "cpu"], [0, [128, 0, 0]], [1.5, 2], [0.5, 0.6], [0.4, 0.6])
def test_paste_op(device, fill_value, ratio, paste_x, paste_y):

    @pipeline_def(batch_size=2, num_threads=1, device_id=0)
    def paste_pipeline():
        images, _ = fn.readers.file(file_list=file_list)
        images = fn.decoders.image(images, device="mixed" if device == "gpu" else "cpu")
        pasted_images = fn.paste(
            images, ratio=ratio, paste_x=paste_x, paste_y=paste_y, fill_value=fill_value
        )
        return images, pasted_images

    pipe = paste_pipeline()
    pipe.build()
    output = pipe.run()

    for orig, pasted in zip(output[0], output[1]):
        expect_im = opencv_paste(np.array(orig.as_cpu()), ratio, paste_x, paste_y, fill_value)
        test_im = np.array(pasted.as_cpu())
        assert np.array_equal(test_im, expect_im)


@raises(RuntimeError, "ratio of less than 1 is not supported")
def test_paste_op_invalid_ratio():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def paste_pipeline():
        images, _ = fn.readers.file(file_list=file_list)
        images = fn.decoders.image(images, device="mixed")
        return fn.paste(images, ratio=0.5, fill_value=0)

    pipe = paste_pipeline()
    pipe.build()
    pipe.run()


@raises(ValueError, "The number of dimensions 4 does not match any of the allowed layouts")
def test_paste_op_invalid_dimension():

    def bad_datasrc():
        yield np.empty((5, 6, 7, 8), dtype=np.uint8)

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def paste_pipeline():
        data = fn.external_source(source=bad_datasrc, batch=False, device="cpu")
        return fn.paste(data, ratio=2, fill_value=0)

    pipe = paste_pipeline()
    pipe.build()
    pipe.run()
