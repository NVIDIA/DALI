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

from nose2.tools import params
from nvidia.dali import fn, types, pipeline_def
import numpy as np
from test_utils import check_batch, get_dali_extra_path

test_data_root = get_dali_extra_path()
db_2d_folder = os.path.join(test_data_root, "db", "lmdb")
db_vid_folder = os.path.join(test_data_root, "db", "video", "sintel", "video_files")


@pipeline_def(num_threads=4, batch_size=8, device_id=0, seed=1234)
def rcm_pipe(device, mode, roi_start=None, roi_end=None):
    roi_relative = True if roi_start or roi_end else None
    files, labels = fn.readers.caffe(path=db_2d_folder, random_shuffle=True)
    images = fn.decoders.image(files, device="mixed" if device == "gpu" else "cpu")
    flip_x = fn.random.coin_flip(dtype=types.INT32)
    flip_y = fn.random.coin_flip(dtype=types.INT32)
    flip = flip_x | (flip_y * 2)
    if mode == "not_larger":  # avoid invalid crops
        size = fn.random.uniform(range=(800, 1000), shape=(2,), dtype=types.FLOAT)
    else:
        size = fn.random.uniform(range=(224, 480), shape=(2,), dtype=types.FLOAT)
    crop_w = fn.random.uniform(range=(100, 224), dtype=types.FLOAT)
    crop_h = fn.random.uniform(range=(100, 224), dtype=types.FLOAT)
    crop_x = fn.random.uniform(range=(0, 1))
    crop_y = fn.random.uniform(range=(0, 1))
    out = fn.resize_crop_mirror(
        images,
        size=size,
        mode=mode,
        roi_start=roi_start,
        roi_end=roi_end,
        roi_relative=roi_relative,
        crop_w=crop_w,
        crop_h=crop_h,
        crop_pos_x=crop_x,
        crop_pos_y=crop_y,
        mirror=flip,
    )
    resized = fn.resize(
        images,
        size=size,
        mode=mode,
        roi_start=roi_start,
        roi_end=roi_end,
        roi_relative=roi_relative,
    )
    cropped = fn.crop(resized, crop_w=crop_w, crop_h=crop_h, crop_pos_x=crop_x, crop_pos_y=crop_y)
    flipped = fn.flip(cropped, horizontal=flip_x, vertical=flip_y)
    return out, flipped


@params(
    ("cpu", "not_larger", None, None),
    ("cpu", None, (0.7, 0.2), (0.1, 0.8)),
    ("gpu", "not_smaller", None, None),
    ("gpu", "stretch", (0.3, 0.8), (0.9, 0.1)),
)
def test_vs_separate_ops(dev, mode, roi_start, roi_end):
    pipe = rcm_pipe(dev, mode, roi_start, roi_end)
    for _ in range(5):
        rcm, separate = pipe.run()
        check_batch(rcm, separate, len(rcm), 1e-3, 1)


@pipeline_def(num_threads=1, batch_size=1, device_id=0, seed=1234)
def rcm_video_pipe(source_name, device, mode, height, width, roi_start, roi_end, crop):
    # Read the input video
    encoded_video = fn.external_source(
        device="cpu",
        name=source_name,
        no_copy=False,
        blocking=True,
        dtype=types.UINT8,
    )
    decoded = fn.experimental.decoders.video(
        encoded_video, device="mixed", start_frame=0, sequence_length=30
    )

    # Resize the video to 1280x720
    decoded = fn.resize_crop_mirror(
        decoded, size=(height, width), roi_start=roi_start, roi_end=roi_end, crop=crop, mirror=2
    )

    return decoded


@params(
    ("cpu", None, 720, 1280, (0.0, 0.0), (1.0, 1.0), (720.0, 1280.0)),
    ("gpu", None, 720, 1280, (0.0, 0.0), (1.0, 1.0), (720.0, 1280.0)),
    ("cpu", None, 720, 1280, (0.0, 0.0), (1.0, 1.0), (200.0, 320.0)),
    ("gpu", None, 720, 1280, (0.0, 0.0), (1.0, 1.0), (200.0, 320.0)),
    ("cpu", None, 720, 1280, (0.0, 0.0), (0.5, 0.5), (720.0, 1280.0)),
    ("gpu", None, 720, 1280, (0.0, 0.0), (0.5, 0.5), (720.0, 1280.0)),
    ("cpu", None, 720, 1280, (0.0, 0.0), (0.5, 0.5), (480.0, 640.0)),
    ("gpu", None, 720, 1280, (0.0, 0.0), (0.5, 0.5), (480.0, 640.0)),
)
def test_video_rcm(dev, mode, height, width, roi_start, roi_end, crop):
    pipe = rcm_video_pipe(
        "encoded_video", dev, mode, height, width, roi_start, roi_end, crop, prefetch_queue_depth=1
    )

    for i, file_name in enumerate(os.listdir(db_vid_folder)):
        file_path = os.path.join(db_vid_folder, file_name)
        decoded = pipe.run(
            encoded_video=np.expand_dims(np.fromfile(file_path, dtype=np.uint8), axis=0)
        )
        v_shape = decoded[0].shape()[0][1:3]
        assert v_shape == crop
