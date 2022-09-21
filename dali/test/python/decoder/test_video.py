# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import numpy as np
import cv2
import nvidia.dali.types as types
from os import environ
import glob

DALI_EXTRA = environ['DALI_EXTRA_PATH']

filenames = glob.glob(f'{DALI_EXTRA}/db/video/[cv]fr/*.mp4')


@pipeline_def(device_id=0)
def video_decoder_pipeline(source):
    data = fn.external_source(source=source, dtype=types.UINT8, ndim=1)
    return fn.experimental.decoders.video(data)


def video_length(filename):
    cap = cv2.VideoCapture(filename)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def reference_pipeline(filename):
    seq_length = video_length(filename)
    return fn.experimental.readers.video(filenames=[filename], sequence_length=seq_length)


def video_loader(batch_size, epochs):
    idx = 0
    while idx < epochs * len(filenames):
        batch = []
        for _ in range(batch_size):
            batch.append(np.fromfile(filenames[idx % len(filenames)], dtype=np.uint8))
            idx = idx + 1
        yield batch


def video_decoder_iter(batch_size, epochs=1):
    pipe = video_decoder_pipeline(batch_size=batch_size, device_id=0, num_threads=4,
                                  source=video_loader(batch_size, epochs))
    pipe.build()
    for _ in range(int((epochs * len(filenames) + batch_size - 1) / batch_size)):
        output, = pipe.run()
        for i in range(batch_size):
            yield np.array(output[i])


def ref_iter(epochs=1):
    for _ in range(epochs):
        for filename in filenames:
            pipe = reference_pipeline(filename)
            pipe.build()
            output, = pipe.run()
            yield np.array(output[0])


def test_video_decoder_cpu():
    batch_size = 4
    for seq, ref_seq in zip(video_decoder_iter(batch_size), ref_iter()):
        assert seq.shape == ref_seq.shape
        assert np.array_equal(seq, ref_seq)
