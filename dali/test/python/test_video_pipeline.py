# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.pipeline import Pipeline, pipeline_def
import nvidia.dali.fn as fn
import numpy as np
import tempfile
from test_utils import check_output_pattern, skip_if_m60

from nose_utils import assert_raises, raises

VIDEO_DIRECTORY = "/tmp/video_files"
PLENTY_VIDEO_DIRECTORY = "/tmp/many_video_files"
VIDEO_FILES = os.listdir(VIDEO_DIRECTORY)
PLENTY_VIDEO_FILES = os.listdir(PLENTY_VIDEO_DIRECTORY)
VIDEO_FILES = [VIDEO_DIRECTORY + "/" + f for f in VIDEO_FILES]
PLENTY_VIDEO_FILES = [PLENTY_VIDEO_DIRECTORY + "/" + f for f in PLENTY_VIDEO_FILES]
FILE_LIST = "/tmp/file_list.txt"
MUTLIPLE_RESOLUTION_ROOT = "/tmp/video_resolution/"

test_data_root = get_dali_extra_path()
video_data_root = os.path.join(test_data_root, "db", "video")
corrupted_video_data_root = os.path.join(video_data_root, "corrupted")
video_containers_data_root = os.path.join(test_data_root, "db", "video", "containers")
video_types = ["avi", "mov", "mkv", "mpeg"]

ITER = 6
BATCH_SIZE = 4
COUNT = 5


class VideoPipe(Pipeline):
    def __init__(
        self,
        batch_size,
        data,
        shuffle=False,
        stride=1,
        step=-1,
        device_id=0,
        num_shards=1,
        dtype=types.FLOAT,
        sequence_length=COUNT,
    ):
        super().__init__(batch_size, num_threads=2, device_id=device_id, seed=12)
        self.input = ops.readers.Video(
            device="gpu",
            filenames=data,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=num_shards,
            random_shuffle=shuffle,
            normalized=True,
            image_type=types.YCbCr,
            dtype=dtype,
            step=step,
            stride=stride,
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output


class VideoPipeList(Pipeline):
    def __init__(
        self,
        batch_size,
        data,
        device_id=0,
        sequence_length=COUNT,
        step=-1,
        stride=1,
        file_list_frame_num=True,
        file_list_include_preceding_frame=False,
        skip_vfr_check=False,
    ):
        super().__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.readers.Video(
            device="gpu",
            file_list=data,
            sequence_length=sequence_length,
            step=step,
            stride=stride,
            file_list_frame_num=file_list_frame_num,
            file_list_include_preceding_frame=file_list_include_preceding_frame,
            skip_vfr_check=skip_vfr_check,
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output


class VideoPipeRoot(Pipeline):
    def __init__(self, batch_size, data, device_id=0, sequence_length=COUNT):
        super().__init__(batch_size, num_threads=2, device_id=device_id)
        self.input = ops.readers.Video(
            device="gpu", file_root=data, sequence_length=sequence_length, random_shuffle=True
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output


def test_simple_videopipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES)
    for i in range(ITER):
        out = pipe.run()
        assert out[0].layout() == "FHWC"
    del pipe


def test_wrong_length_sequence_videopipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, sequence_length=100000)
    with assert_raises(RuntimeError, glob="There are no valid sequences in the provided dataset"):
        pipe.build()


def check_videopipeline_supported_type(dtype):
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, dtype=dtype)
    for i in range(ITER):
        _ = pipe.run()
    del pipe


@raises(RuntimeError, glob="Data type must be FLOAT or UINT8")
def check_videopipeline_unsupported_type(dtype):
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, dtype=dtype)
    pipe.build()


SUPPORTED_TYPES = [types.DALIDataType.FLOAT, types.DALIDataType.UINT8]
ALL_TYPES = list(types.DALIDataType.__members__.values())


def test_simple_videopipeline_supported_types():
    for type in SUPPORTED_TYPES:
        yield check_videopipeline_supported_type, type


def test_simple_videopipeline_not_supported_types():
    for type in set(ALL_TYPES) - set(SUPPORTED_TYPES):
        yield check_videopipeline_unsupported_type, type


def test_file_list_videopipeline():
    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=FILE_LIST)
    for i in range(ITER):
        _ = pipe.run()
    del pipe


def _test_file_list_starts_videopipeline(start, end):
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    list_file.write("{} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0))
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name, sequence_length=1)
    reference_seq_num = pipe.reader_meta("Reader")["epoch_size"]
    del pipe
    os.remove(list_file.name)

    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    if end is None:
        list_file.write("{} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start))
    else:
        list_file.write(
            "{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start, end)
        )
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name, sequence_length=1)
    seq_num = pipe.reader_meta("Reader")["epoch_size"]

    expected_seq_num = reference_seq_num
    if start > 0:
        expected_seq_num -= start
    elif start < 0:
        expected_seq_num = -start

    if end is not None:
        if end > 0:
            expected_seq_num -= reference_seq_num - end
        elif end < 0:
            expected_seq_num += end

    assert expected_seq_num == seq_num, "Reference is {}, expected is {}, obtained {}".format(
        reference_seq_num, expected_seq_num, seq_num
    )
    os.remove(list_file.name)


def test_file_list_starts_ends_videopipeline():
    ranges = [[0, None], [1, None], [0, -1], [2, None], [0, -2], [0, 1], [-1, None], [-3, -1]]
    for r in ranges:
        yield _test_file_list_starts_videopipeline, r[0], r[1]


def test_source_info():
    files = []
    for cont in video_types:
        path = os.path.join(video_containers_data_root, cont)
        files += [path + "/" + f for f in os.listdir(path)]
    files = sorted(files)
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    for f in files:
        list_file.write("{} {} {} {}\n".format(f, 0, 0, 1))
    list_file.close()

    pipe = VideoPipeList(
        batch_size=BATCH_SIZE, data=list_file.name, sequence_length=1, skip_vfr_check=True
    )
    samples_read = 0
    while samples_read < len(files):
        o = pipe.run()
        for idx, t in enumerate(o[0]):
            assert t.source_info() == files[(samples_read + idx) % len(files)]
        samples_read += BATCH_SIZE

    os.remove(list_file.name)


def _create_file_list_include_preceding_frame_pipe(file_list_include_preceding_frame):
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    # make sure that this is close enough to show only one frame
    list_file.write(
        "{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, 0.111, 0.112)
    )
    list_file.close()

    pipe = VideoPipeList(
        batch_size=BATCH_SIZE,
        data=list_file.name,
        sequence_length=1,
        file_list_frame_num=False,
        file_list_include_preceding_frame=file_list_include_preceding_frame,
    )

    return pipe, list_file.name


def test_file_list_include_preceding_frame():
    pipe, list_file_name = _create_file_list_include_preceding_frame_pipe(True)
    pipe.build()

    os.remove(list_file_name)
    for _ in range(3):
        pipe.run()
    seq_num = pipe.reader_meta("Reader")["epoch_size"]

    assert seq_num == 1, "Expected to get only 1 sample, received {}".format(seq_num)


def test_file_list_include_preceding_frame_fail():
    pipe, list_file_name = _create_file_list_include_preceding_frame_pipe(False)

    # there should be no valid sequences
    expected_msg = "Start time number should be lesser or equal to end time for a file"
    with assert_raises(RuntimeError, glob=expected_msg):
        pipe.build()
    os.remove(list_file_name)


def _test_file_list_invalid_range(start, end):
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    if end is None:
        list_file.write("{} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start))
    else:
        list_file.write(
            "{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, start, end)
        )
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name)
    expected_msg = "Start frame number should be lesser or equal to end frame number for a file"
    with assert_raises(RuntimeError, glob=expected_msg):
        pipe.build()
    os.remove(list_file.name)


def test_file_list_invalid_range():
    invalid_ranges = [[-1, 1], [1000000, None], [0, -1000]]
    for r in invalid_ranges:
        yield _test_file_list_invalid_range, r[0], r[1]


def test_file_list_empty_range():
    files = sorted(os.listdir(VIDEO_DIRECTORY))
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    list_file.write("{} {} {} {}\n".format(os.path.join(VIDEO_DIRECTORY, files[0]), 0, 10, 10))
    list_file.close()

    pipe = VideoPipeList(batch_size=BATCH_SIZE, data=list_file.name)
    with assert_raises(RuntimeError, glob="No files were read"):
        pipe.build()
    os.remove(list_file.name)


def test_step_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, step=1)
    for i in range(ITER):
        _ = pipe.run()
    del pipe


def test_stride_video_pipeline():
    pipe = VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, stride=3)
    for i in range(ITER):
        _ = pipe.run()
    del pipe


def test_multiple_resolution_videopipeline():
    pipe = VideoPipeRoot(batch_size=BATCH_SIZE, data=MUTLIPLE_RESOLUTION_ROOT)
    try:
        for i in range(ITER):
            _ = pipe.run()
    except Exception as e:
        if str(e) == "Decoder reconfigure feature not supported":
            print("Multiple resolution test skipped")
        else:
            raise
    del pipe


def test_multi_gpu_video_pipeline():
    gpus = get_gpu_num()
    pipes = [
        VideoPipe(batch_size=BATCH_SIZE, data=VIDEO_FILES, device_id=d, num_shards=gpus)
        for d in range(gpus)
    ]
    for p in pipes:
        p.run()


def test_plenty_of_video_files():
    """
    checks if the readers.Video can handle more than OS max open file limit of opened files at once
    """
    # make sure that there is one sequence per video file
    pipe = VideoPipe(
        batch_size=BATCH_SIZE, data=PLENTY_VIDEO_FILES, step=1000000, sequence_length=1
    )
    iters = math.ceil(len(os.listdir(PLENTY_VIDEO_DIRECTORY)) / BATCH_SIZE)
    for i in range(iters):
        pipe.run()


def check_corrupted_videos(reader_op, reader_args, corrupted_video, msg):
    good_video_path = os.path.join(video_data_root, "cfr", "test_2.mp4")
    corrupted_video_path = os.path.join(corrupted_video_data_root, corrupted_video)
    reader_args["filenames"] = [corrupted_video_path, good_video_path]
    reader_args["sequence_length"] = 3
    pipe = Pipeline(batch_size=BATCH_SIZE, num_threads=4, device_id=0)
    with pipe:
        vid = reader_op(**reader_args)
        pipe.set_outputs(vid)
    with check_output_pattern(pattern=msg):
        pipe.run()


def test_corrupted_videos():
    corrupted_videos = [
        ("moov_atom_not_found.mp4", "Failed to open video file"),
        ("audio_only.mp4", "Could not find a valid video stream in a file"),
    ]

    reader_opts = [
        (fn.readers.video, {"device": "gpu"}),
        (fn.experimental.readers.video, {"device": "gpu"}),
        (fn.experimental.readers.video, {"device": "cpu"}),
    ]
    for reader_op, reader_args in reader_opts:
        for wrong_video, msg in corrupted_videos:
            yield check_corrupted_videos, reader_op, reader_args, wrong_video, msg


def check_container(cont):
    pipe = Pipeline(batch_size=1, num_threads=4, device_id=0)
    path = os.path.join(video_containers_data_root, cont)
    test_videos = [path + "/" + f for f in os.listdir(path)]
    with pipe:
        # mkv container for some reason fails in DALI VFR heuristics
        vid = fn.readers.video(
            device="gpu",
            filenames=test_videos,
            sequence_length=10,
            skip_vfr_check=True,
            stride=1,
            name="Reader",
        )
        pipe.set_outputs(vid)

    iter_num = pipe.reader_meta("Reader")["epoch_size"]
    for _ in range(iter_num):
        pipe.run()


def test_container():
    for cont in video_types:
        yield check_container, cont


def test_pad_sequence():
    def get_epoch_size(pipe):
        meta = pipe.reader_meta()
        return list(meta.values())[0]["epoch_size"]

    @pipeline_def(batch_size=1, num_threads=4, device_id=0)
    def create_video_pipe(filenames, sequence_length=1, stride=1, step=-1, pad_sequences=False):
        fr, lab, fr_num, time_stamp = fn.readers.video(
            device="gpu",
            filenames=filenames,
            labels=[],
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            enable_timestamps=True,
            enable_frame_num=True,
            random_shuffle=False,
            skip_vfr_check=True,
            step=step,
            stride=stride,
            pad_last_batch=True,
            pad_sequences=pad_sequences,
        )
        return fr, lab, fr_num, time_stamp

    video_filename = [
        os.path.join(video_data_root, "sintel", "video_files", "sintel_trailer-720p_2.mp4")
    ]
    dali_pipe = create_video_pipe(video_filename)
    total_number_of_frames = get_epoch_size(dali_pipe)

    sequence_length = 4
    stride = sequence_length // 2
    batch_size = 2
    # second sequence should have only half of the frames
    step = total_number_of_frames - (stride * sequence_length // 2 - 1)
    dali_pipe = create_video_pipe(
        batch_size=batch_size,
        filenames=video_filename,
        sequence_length=sequence_length,
        stride=stride,
        step=step,
        pad_sequences=True,
    )
    assert get_epoch_size(dali_pipe) == 2

    last_sample_frame_count = 1 + (total_number_of_frames - 1 - step) // stride
    assert last_sample_frame_count < sequence_length

    out = dali_pipe.run()
    padded_sampl = 1
    # check padded sample
    # non padded frames should not be 0
    assert np.any(np.array(out[0].as_cpu()[padded_sampl])[0:last_sample_frame_count]) != 0
    # while padded one only 0
    assert np.all(np.array(out[0].as_cpu()[padded_sampl])[last_sample_frame_count + 1 :]) == 0
    assert np.array(out[2].as_cpu()[padded_sampl]) == step
    # non padded samples should have non negative timestamps
    non_padded = np.array(out[3].as_cpu()[padded_sampl])[0:last_sample_frame_count]
    assert np.all(non_padded != np.array([-1] * last_sample_frame_count))
    # while padded one only -1
    padded = np.array(out[3].as_cpu()[padded_sampl])[last_sample_frame_count + 1 :]
    assert np.all(padded == np.array([-1] * (sequence_length - last_sample_frame_count)))

    dali_pipe = create_video_pipe(
        batch_size=2,
        filenames=video_filename,
        sequence_length=sequence_length,
        stride=stride,
        step=step,
        pad_sequences=False,
    )
    # when sequence padding if off we should get only one valid sample in the epoch
    assert get_epoch_size(dali_pipe) == 1

    def divisor_generator(n, max_val):
        for i in range(max_val + 1, 1, -1):
            if n % i == 0:
                return i

    dali_pipe = create_video_pipe(
        batch_size=1, filenames=video_filename, sequence_length=1, stride=1, pad_sequences=False
    )

    # to speed things up read as close as 30 frames at the time, but in the way that the sequences
    # cover the whole video (without padding)
    ref_sequence_length = divisor_generator(get_epoch_size(dali_pipe), 30)

    # extract frames from the test video without padding and compare with one from padded pipeline
    dali_pipe = create_video_pipe(
        batch_size=1,
        filenames=video_filename,
        sequence_length=ref_sequence_length,
        stride=1,
        pad_sequences=False,
    )
    ts_index = 0
    sampl_idx = 0
    for _ in range(get_epoch_size(dali_pipe)):
        ref_out = dali_pipe.run()
        # run over all frame timestamps and compare them with one from the tested pipeline
        for ref_idx in range(ref_sequence_length):
            # if we get into padded samples break
            if np.array(out[3].as_cpu()[sampl_idx])[ts_index] == -1:
                break
            # if there is a match compare frames itself and move to next timestamp/sample
            # from the tested batch
            sample_stamp = np.array(out[3].as_cpu()[sampl_idx])
            ref_sample_stamp = np.array(ref_out[3].as_cpu()[0])
            if sample_stamp[ts_index] == ref_sample_stamp[ref_idx]:
                sample = np.array(out[0].as_cpu()[sampl_idx])
                sample_ref = np.array(ref_out[0].as_cpu()[0])
                assert np.all(sample[ts_index] == sample_ref[ref_idx])
                ts_index += 1
                if ts_index == sequence_length:
                    ts_index = 0
                    sampl_idx += 1
                # it should break earlier and not get here at all, as we expect to have padded
                # sample in the tested pipeline
                if sampl_idx == batch_size:
                    assert False

    assert sampl_idx == padded_sampl
    assert ts_index == last_sample_frame_count


def test_10bit_vid_reconfigure():
    skip_if_m60()
    batch_size = 1
    sequence_length = 5
    iter = 5
    filenames = [
        os.path.join(video_data_root, "hevc", "sintel_trailer-720p.mp4"),
        os.path.join(video_data_root, "hevc", "sintel_trailer-720p_10b.mp4"),
    ]
    filenames2 = [
        os.path.join(video_data_root, "hevc", "sintel_trailer-720p_10b.mp4"),
        os.path.join(video_data_root, "hevc", "sintel_trailer-720p.mp4"),
    ]

    @pipeline_def(device_id=0)
    def video_decoder_pipeline():
        a = fn.readers.video(
            filenames=filenames,
            sequence_length=sequence_length,
            device="gpu",
            random_shuffle=True,
            seed=1234,
        )
        b = fn.readers.video(
            filenames=filenames2,
            sequence_length=sequence_length,
            device="gpu",
            random_shuffle=True,
            seed=1234,
        )
        return a, b

    pipe = video_decoder_pipeline(batch_size=batch_size, num_threads=1, device_id=0)
    for _ in range(iter):
        (a, b) = pipe.run()
        a = a.as_cpu().as_array()
        b = b.as_cpu().as_array()
        absdiff = np.abs(a.astype(int) - b.astype(int))
        assert np.mean(absdiff) < 2, f"error is {np.mean(absdiff)}"


def test_2gb_sequence():
    skip_if_m60()
    # make sure the sequence exceeds beyond 2GB
    sequence_length = int(math.ceil((1 << 31) / 720 / 1280 / 3) + 3)

    @pipeline_def
    def video_pipe():
        video, label = fn.readers.video(
            device="gpu",
            sequence_length=sequence_length,
            filenames=os.path.join(video_data_root, "hevc", "sintel_trailer-720p.mp4"),
            labels=[],
        )
        return video, label

    batch_size = 1
    pipe = video_pipe(batch_size=batch_size, num_threads=1, device_id=0)
    for _ in range(2):
        pipe.run()
