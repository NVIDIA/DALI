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
import glob
from test_utils import get_dali_extra_path
from nvidia.dali.backend import TensorListGPU
from nose2.tools import params

filenames = glob.glob(f'{get_dali_extra_path()}/db/video/[cv]fr/*.mp4')
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: 'hevc' not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator itself
filenames = filter(lambda filename: 'mpeg4' not in filename, filenames)

files = [np.fromfile(
    filename, dtype=np.uint8) for filename in filenames]


@pipeline_def(device_id=0)
def video_decoder_pipeline(source, device='cpu'):
    data = fn.external_source(source=source, dtype=types.UINT8, ndim=1)
    return fn.experimental.decoders.video(data, device=device)


def video_length(filename):
    cap = cv2.VideoCapture(filename)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def reference_pipeline(filename, device='cpu'):
    seq_length = video_length(filename)
    return fn.experimental.readers.video(filenames=[filename], sequence_length=seq_length,
                                         device=device)


def video_loader(batch_size, epochs):
    idx = 0
    while idx < epochs * len(files):
        batch = []
        for _ in range(batch_size):
            batch.append(files[idx % len(files)])
            idx = idx + 1
        yield batch


def video_decoder_iter(batch_size, epochs=1, device='cpu'):
    pipe = video_decoder_pipeline(batch_size=batch_size, device_id=0, num_threads=4,
                                  source=video_loader(batch_size, epochs), device=device)
    pipe.build()
    for _ in range(int((epochs * len(files) + batch_size - 1) / batch_size)):
        output, = pipe.run()
        if isinstance(output, TensorListGPU):
            output = output.as_cpu()
        for i in range(batch_size):
            yield np.array(output[i])


def ref_iter(epochs=1, device='cpu'):
    for _ in range(epochs):
        for filename in filenames:
            pipe = reference_pipeline(filename, device=device)
            pipe.build()
            output, = pipe.run()
            if isinstance(output, TensorListGPU):
                output = output.as_cpu()
            yield np.array(output[0])


@params('cpu', 'mixed')
def test_video_decoder(device):
    batch_size = 3
    epochs = 3
    decoder_iter = video_decoder_iter(batch_size, epochs, device)
    ref_dec_iter = ref_iter(epochs, device='cpu' if device == 'cpu' else 'gpu')
    for seq, ref_seq in zip(decoder_iter, ref_dec_iter):
        assert seq.shape == ref_seq.shape
        assert np.array_equal(seq, ref_seq)
