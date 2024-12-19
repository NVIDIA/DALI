# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import cv2
import numpy as np

import nvidia.dali.fn.plugin.video as video

from test_utils import get_dali_extra_path


@pipeline_def(device_id=0, num_threads=4, batch_size=1)
def video_decoder_pipeline(data_path):
    data = fn.external_source(
        source=lambda: [
            np.fromfile(data_path, np.uint8),
        ],
        dtype=types.UINT8,
    )
    return video.decoder(data, device="mixed", end_frame=50)


def run_video_decoding_test(test_file_path, frame_list_file_path, frames_directory_path):
    pipeline = video_decoder_pipeline(test_file_path)

    (output,) = pipeline.run()
    frames = output.as_cpu().as_array()

    with open(frame_list_file_path) as f:
        frame_list = f.read().splitlines()

    for i, frame in enumerate(frames[0]):
        # Check if the frame is equal to the ground truth frame.
        # Due to differences in how the decoding is implemented in
        # different video codecs, we can't guarantee that the frames
        # will be exactly the same. Main purpose of this test is to
        # check if the decoding is working and we hit the correct frames.
        ground_truth = cv2.imread(f"{frames_directory_path}/{frame_list[i]}")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not np.average(frame - ground_truth) < 15:
            assert False, f"Frame {i} is not equal"


def test_cfr_h264_mp4_decoding():
    run_video_decoding_test(
        get_dali_extra_path() + "/db/video/cfr/test_1.mp4",
        f"{get_dali_extra_path()}/db/video/cfr/frames_1/frames_list.txt",
        f"{get_dali_extra_path()}/db/video/cfr/frames_1",
    )


def test_cfr_h265_mp4_decoding():
    run_video_decoding_test(
        get_dali_extra_path() + "/db/video/cfr/test_1_hevc.mp4",
        f"{get_dali_extra_path()}/db/video/cfr/frames_1/frames_list.txt",
        f"{get_dali_extra_path()}/db/video/cfr/frames_1",
    )


def test_cfr_av1_mp4_decoding():
    try:
        run_video_decoding_test(
            get_dali_extra_path() + "/db/video/cfr/test_1_av1.mp4",
            f"{get_dali_extra_path()}/db/video/cfr/frames_1/frames_list.txt",
            f"{get_dali_extra_path()}/db/video/cfr/frames_1",
        )
        return
    except Exception as e:
        assert "Codec not supported on this GPU" in str(e), "Unexpected error message: {}".format(e)
        return


def test_vfr_h264_mp4_decoding():
    run_video_decoding_test(
        get_dali_extra_path() + "/db/video/vfr/test_1.mp4",
        f"{get_dali_extra_path()}/db/video/vfr/frames_1/frames_list.txt",
        f"{get_dali_extra_path()}/db/video/vfr/frames_1",
    )


def test_vfr_hevc_mp4_decoding():
    run_video_decoding_test(
        get_dali_extra_path() + "/db/video/vfr/test_1_hevc.mp4",
        f"{get_dali_extra_path()}/db/video/vfr/frames_1_hevc/frames_list.txt",
        f"{get_dali_extra_path()}/db/video/vfr/frames_1_hevc",
    )


def test_vfr_av1_mp4_decoding():
    try:
        run_video_decoding_test(
            get_dali_extra_path() + "/db/video/vfr/test_1_av1.mp4",
            f"{get_dali_extra_path()}/db/video/vfr/frames_1/frames_list.txt",
            f"{get_dali_extra_path()}/db/video/vfr/frames_1",
        )
        return
    except Exception as e:
        assert "Codec not supported on this GPU" in str(e), "Unexpected error message: {}".format(e)
        return
