# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function

import os
import random
from math import ceil, sqrt

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.backend_impl import TensorListGPU
from nvidia.dali.pipeline import Pipeline

VIDEO_DIRECTORY="video_files"
VIDEO_FILES=os.listdir(VIDEO_DIRECTORY)
VIDEO_FILES = [VIDEO_DIRECTORY + '/' + f for f in VIDEO_FILES]

ITER=6
BATCH_SIZE=4
COUNT=5

class VideoPipe(Pipeline):
    def __init__(self, batch_size, data, shuffle=False, stride=1, step=-1):
        super(VideoPipe, self).__init__(batch_size, num_threads=2, device_id=0, seed=12)
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=COUNT,
                                     shard_id=0, num_shards=1, random_shuffle=shuffle,
                                     normalized=True, image_type=types.YCbCr, dtype=types.FLOAT,
                                     step=step, stride=stride)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

def test_simple_videopipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        pipe_out = pipe.run()
    del pipe

def test_step_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, step=1)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        pipe_out = pipe.run()
    del pipe

def test_stride_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, stride=3)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        pipe_out = pipe.run()
    del pipe

