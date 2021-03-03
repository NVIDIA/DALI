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
import os
import math
from test_utils import get_gpu_num
from test_utils import get_dali_extra_path

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.backend_impl import TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import re
import tempfile

from nose.tools import assert_raises, raises

VIDEO_DIRECTORY = "/tmp/video_files"
PLENTY_VIDEO_DIRECTORY = "/tmp/many_video_files"
VIDEO_FILES = os.listdir(VIDEO_DIRECTORY)
PLENTY_VIDEO_FILES=  os.listdir(PLENTY_VIDEO_DIRECTORY)
VIDEO_FILES = [VIDEO_DIRECTORY + '/' + f for f in VIDEO_FILES]
PLENTY_VIDEO_FILES = [PLENTY_VIDEO_DIRECTORY + '/' + f for f in PLENTY_VIDEO_FILES]
FILE_LIST = "/tmp/file_list.txt"
MUTLIPLE_RESOLUTION_ROOT = '/tmp/video_resolution/'

test_data_root = os.environ['DALI_EXTRA_PATH']
video_data_root = os.path.join(test_data_root, 'db', 'video')
corrupted_video_data_root = os.path.join(video_data_root, 'corrupted')
video_containers_data_root = os.path.join(test_data_root, 'db', 'video', 'containers')
video_types = ['avi', 'mov', 'mkv', 'mpeg']

ITER=6
BATCH_SIZE=4
COUNT=5


class VideoPipe(Pipeline):
    def __init__(self, batch_size, data, shuffle=False, stride=1, step=-1, device_id=0, num_shards=1,
                 dtype=types.FLOAT, sequence_length=COUNT):
        super(VideoPipe, self).__init__(batch_size, num_threads=2, device_id=device_id, seed=12)
        self.input = ops.readers.Video(device="gpu", filenames=data, sequence_length=sequence_length,
                                       shard_id=0, num_shards=num_shards, random_shuffle=shuffle,
                                       normalized=True, image_type=types.YCbCr, dtype=dtype,
                                       step=step, stride=stride)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

class VideoPipeList(Pipeline):
    def __init__(self, batch_size, data, device_id=0, sequence_length=COUNT, step=-1, stride=1):
        super(VideoPipeList, self).__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.readers.Video(device="gpu", file_list=data, sequence_length=sequence_length,
                                       step=step, stride=stride, file_list_frame_num=True)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

class VideoPipeRoot(Pipeline):
    def __init__(self, batch_size, data, device_id=0, sequence_length=COUNT):
        super(VideoPipeRoot, self).__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.readers.Video(device="gpu", file_root=data, sequence_length=sequence_length,
                                       random_shuffle=True)

    def define_graph(self):
        output = self.input(name="Reader")
        return output

def test_simple_videopipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        out = pipe.run()
        assert(out[0].layout() == "FHWC")
    del pipe

def test_wrong_length_sequence_videopipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, sequence_length=100000)
    assert_raises(RuntimeError, pipe.build)

def check_videopipeline_supported_type(dtype):
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, dtype=dtype)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        _ = pipe.run()
    del pipe

SUPPORTED_TYPES = [types.DALIDataType.FLOAT, types.DALIDataType.UINT8]
ALL_TYPES = list(types.DALIDataType.__members__.values())

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
        _ = pipe.run()
    del pipe

def _test_file_list_starts_videopipeline(start, end):
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    list_file.write("{} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0))
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name, sequence_length=1)
    pipe.build()
    reference_seq_num = pipe.reader_meta("Reader")["epoch_size"]
    del pipe
    os.remove(list_file.name)

    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    if end is None:
        list_file.write("{} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start))
    else:
        list_file.write("{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start, end))
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name, sequence_length=1)
    pipe.build()
    seq_num = pipe.reader_meta("Reader")["epoch_size"]

    expected_seq_num = reference_seq_num
    if start > 0:
        expected_seq_num -= start
    elif start < 0:
        expected_seq_num = -start

    if end is not None:
        if end > 0:
            expected_seq_num -= (reference_seq_num - end)
        elif end < 0:
            expected_seq_num += end

    assert expected_seq_num == seq_num, "Reference is {}, expected is {}, obtained {}".format(reference_seq_num, expected_seq_num, seq_num)
    os.remove(list_file.name)

def test_file_list_starts_ends_videopipeline():
    ranges = [
        [0, None],
        [1, None],
        [0, -1],
        [2, None],
        [0, -2],
        [0, 1],
        [-1, None],
        [-3, -1]
    ]
    for r in ranges:
        yield _test_file_list_starts_videopipeline, r[0], r[1]

def _test_file_list_empty_videopipeline(start, end):
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    if end is None:
        list_file.write("{} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start))
    else:
        list_file.write("{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start, end))
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name)
    assert_raises(RuntimeError, pipe.build)
    os.remove(list_file.name)

def test_file_list_empty_videopipeline():
    invalid_ranges = [
        [0, 0],
        [10, 10],
        [-1, 1],
        [1000000, None],
        [0, -1000]
    ]
    for r in invalid_ranges:
        yield _test_file_list_empty_videopipeline, r[0], r[1]

def test_step_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, step=1)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        _ = pipe.run()
    del pipe

def test_stride_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, stride=3)
    pipe.build()
    for i in range(ITER):
        print("Iter " + str(i))
        _ = pipe.run()
    del pipe

def test_multiple_resolution_videopipeline():
    pipe = VideoPipeRoot(batch_size=BATCH_SIZE, data=MUTLIPLE_RESOLUTION_ROOT)
    try:
        pipe.build()
        for i in range(ITER):
            print("Iter " + str(i))
            _ = pipe.run()
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

# checks if the readers.Video can handle more than OS max open file limit of opened files at once
def test_plenty_of_video_files():
    # make sure that there is one sequence per video file
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=PLENTY_VIDEO_FILES, step=1000000, sequence_length=1)
    pipe.build()
    iters = math.ceil(len(os.listdir(PLENTY_VIDEO_DIRECTORY)) / BATCH_SIZE)
    for i in range(iters):
        print("Iter " + str(i))
        pipe.run()

@raises(RuntimeError)
def check_corrupted_videos():
    corrupted_videos = [corrupted_video_data_root + '/' + f for f in os.listdir(corrupted_video_data_root)]
    for corrupted in corrupted_videos:
        pipe = Pipeline(batch_size=BATCH_SIZE, num_threads=4, device_id=0)
        with pipe:
            vid = fn.readers.video(device="gpu", filenames=corrupted, sequence_length=1)
            pipe.set_outputs(vid)
        pipe.build()

def test_corrupted_videos():
    check_corrupted_videos()

def check_container(cont):
    pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
    path = os.path.join(video_containers_data_root, cont)
    test_videos = [path + '/' + f for f in os.listdir(path)]
    with pipe:
        # mkv container for some reason fails in DALI VFR heuristics
        vid = fn.readers.video(device="gpu", filenames=test_videos, sequence_length=10,
                               skip_vfr_check=True, stride=1, name="Reader")
        pipe.set_outputs(vid)
    pipe.build()

    iter_num = pipe.reader_meta("Reader")["epoch_size"]
    for _ in range(iter_num):
        pipe.run()

def check_container():
    for cont in video_types:
        yield test_container, cont
