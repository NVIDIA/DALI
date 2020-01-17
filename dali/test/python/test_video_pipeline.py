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
import math
from test_utils import get_gpu_num
from test_utils import get_dali_extra_path

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.backend_impl import TensorListGPU
from nvidia.dali.pipeline import Pipeline
import re

from nose.tools import assert_raises

VIDEO_DIRECTORY = "/tmp/video_files"
PLENTY_VIDEO_DIRECTORY = "/tmp/many_video_files"
VIDEO_FILES = os.listdir(VIDEO_DIRECTORY)
PLENTY_VIDEO_FILES=  os.listdir(PLENTY_VIDEO_DIRECTORY)
VIDEO_FILES = [VIDEO_DIRECTORY + '/' + f for f in VIDEO_FILES]
PLENTY_VIDEO_FILES = [PLENTY_VIDEO_DIRECTORY + '/' + f for f in PLENTY_VIDEO_FILES]
FILE_LIST = "/tmp/file_list.txt"
MUTLIPLE_RESOLUTION_ROOT = '/tmp/video_resolution/'

ITER=6
BATCH_SIZE=4
COUNT=5


class VideoPipe(Pipeline):
    def __init__(self, batch_size, data, shuffle=False, stride=1, step=-1, device_id=0, num_shards=1,
                 dtype=types.FLOAT, sequence_length=COUNT):
        super(VideoPipe, self).__init__(batch_size, num_threads=2, device_id=device_id, seed=12)
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,
                                     shard_id=0, num_shards=num_shards, random_shuffle=shuffle,
                                     normalized=True, image_type=types.YCbCr, dtype=dtype,
                                     step=step, stride=stride)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

class VideoPipeList(Pipeline):
    def __init__(self, batch_size, data, device_id=0, sequence_length=COUNT):
        super(VideoPipeList, self).__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.VideoReader(device="gpu", file_list=data, sequence_length=sequence_length)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

class VideoPipeRoot(Pipeline):
    def __init__(self, batch_size, data, device_id=0, sequence_length=COUNT):
        super(VideoPipeRoot, self).__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.VideoReader(device="gpu", file_root=data, sequence_length=sequence_length,
                                     random_shuffle=True, initial_fill=10)

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

def check_videopipeline_supported_type(dtype):
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, dtype=dtype)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        pipe_out = pipe.run()
    del pipe

SUPPORTED_TYPES = [types.DALIDataType.FLOAT, types.DALIDataType.UINT8]
ALL_TYPES = [v for k, v in types.DALIDataType.__dict__.items() if not re.match("__(.*)__", str(k))]

def test_simple_videopipeline_supported_types():
    for type in SUPPORTED_TYPES:
        yield check_videopipeline_supported_type, type

def test_simple_videopipeline_not_supported_types():
    for type in set(ALL_TYPES) - set(SUPPORTED_TYPES):
        yield assert_raises, RuntimeError, check_videopipeline_supported_type, type

def test_file_list_videopipeline():
    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=FILE_LIST)
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

def test_multiple_resolution_videopipeline():
    pipe = VideoPipeRoot(batch_size=BATCH_SIZE, data=MUTLIPLE_RESOLUTION_ROOT)
    try:
        pipe.build()
        for i in range(ITER):
            print("Iter " + str(i))
            pipe_out = pipe.run()
    except Exception as e:
        if str(e) == "Decoder reconfigure feature not supported":
            print("Multiple resolution test skipped")
        else:
            raise
    del pipe

def test_multi_gpu_video_pipeline():
    gpus = get_gpu_num()
    pipes = [VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, device_id=d, num_shards=gpus) for d in range(gpus)]
    for p in pipes:
        p.build()
        p.run()

# checks if the VideoReader can handle more than OS max open file limit of opened files at once
def test_plenty_of_video_files():
    # make sure that there is one sequence per video file
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=PLENTY_VIDEO_FILES, step=1000000, sequence_length=1)
    pipe.build()
    iters = math.ceil(len(os.listdir(PLENTY_VIDEO_DIRECTORY)) / BATCH_SIZE)
    for i in range(iters):
        print("Iter " + str(i))
        pipe.run()
