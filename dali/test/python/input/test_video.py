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
import nvidia.dali.types as types
import glob
from test_utils import get_dali_extra_path
from nose2.tools import params
import unittest

filenames = glob.glob(f'{get_dali_extra_path()}/db/video/[cv]fr/*.mp4')

files = [np.fromfile(
    filename, dtype=np.uint8) for filename in filenames]


@pipeline_def
def video_decoder_pipeline(input_name, device='cpu'):
    data = fn.external_source(name=input_name, dtype=types.UINT8, ndim=1)
    vid = fn.experimental.decoders.video(data, device=device)
    # vid = fn.resize(vid, size=(15, 20),interp_type=types.INTERP_NN)
    return vid


@pipeline_def
def video_input_pipeline(input_name, frames_per_sequence, last_sequence_policy='partial',
                         device='cpu'):
    vid = fn.experimental.inputs.video(name=input_name, device=device, blocking=False,
                                       frames_per_sequence=frames_per_sequence)
    # vid = fn.resize(vid, size=(15, 20),interp_type=types.INTERP_NN)
    return vid


common_pipeline_params = {
    'num_threads': 1,
    'device_id': 0,
    'exec_pipelined': False,
    'exec_async': False
}


def get_num_frames(encoded_video):
    decoder_pipe = video_decoder_pipeline(input_name="VIDEO_INPUT", batch_size=1, device="cpu",
                                          **common_pipeline_params)
    decoder_pipe.build()
    decoder_pipe.feed_input("VIDEO_INPUT", [encoded_video])
    decoder_out = decoder_pipe.run()
    return decoder_out[0].as_array()[0].shape[0]


def portion_out_reference_sequence(decoder_pipe_out, frames_per_sequence, batch_size):
    ref_sequence = decoder_pipe_out[0].as_array()[0]
    num_frames = ref_sequence.shape[0]
    if frames_per_sequence * batch_size > num_frames:
        raise ArgumentError(
            "Provided video is too short. Pass longer video to the test or adjust test params (frames_per_sequence and batch_size).")
    n_batches = num_frames // (batch_size * frames_per_sequence)
    ref_sequence = ref_sequence[:n_batches * frames_per_sequence * batch_size]
    sh = ref_sequence.shape
    ref_sequence = ref_sequence.reshape(n_batches, batch_size, frames_per_sequence, sh[1], sh[2],
                                        sh[3])
    for rs in ref_sequence:
        yield rs


@params(("cpu", 5))
def test_video_input_compare_with_video_decoder(device, frames_per_sequence):
    input_name = "VIDEO_INPUT"

    decoder_pipe = video_decoder_pipeline(input_name=input_name, batch_size=1, device=device,
                                          **common_pipeline_params)
    input_pipe = video_input_pipeline(input_name=input_name, batch_size=1,
                                      frames_per_sequence=frames_per_sequence, device=device,
                                      **common_pipeline_params)

    decoder_pipe.build()
    decoder_pipe.feed_input(input_name, [files[0]])
    decoder_out = decoder_pipe.run()

    input_pipe.build()
    input_pipe.feed_input(input_name, np.array([[files[0]]]))
    num_frames = decoder_out[0].as_array()[0].shape[0]

    for seq_idx in range(0, num_frames - frames_per_sequence, frames_per_sequence):
        ref_seq = decoder_out[0].as_array()[0][seq_idx:seq_idx + frames_per_sequence]
        input_out = input_pipe.run()
        test_seq = input_out[0].as_array()[0]
        assert np.all(ref_seq == test_seq)


@params(
    ("cpu", 5, 2),
    ("cpu", 7, 3),
    ("cpu", 2, 7),
)
def test_video_input_compare_with_video_decoder_batch_size(device, frames_per_sequence, batch_size):
    input_name = "VIDEO_INPUT"

    decoder_pipe = video_decoder_pipeline(input_name=input_name, batch_size=1, device=device,
                                          **common_pipeline_params)
    input_pipe = video_input_pipeline(input_name=input_name, batch_size=batch_size,
                                      frames_per_sequence=frames_per_sequence, device=device,
                                      **common_pipeline_params)

    decoder_pipe.build()
    decoder_pipe.feed_input(input_name, [files[0]])
    decoder_out = decoder_pipe.run()

    input_pipe.build()
    input_pipe.feed_input(input_name, np.array([[files[0]]]))

    for ref_seq in portion_out_reference_sequence(decoder_out, frames_per_sequence, batch_size):
        input_out = input_pipe.run()
        test_seq = input_out[0].as_array()
        assert np.all(ref_seq == test_seq)


@params(
    ("cpu", 7),
    # ("cpu", 7),
    # ("cpu", 2),
)
def test_video_input_partial_vs_pad(device, frames_per_sequence):
    input_name = "VIDEO_INPUT"
    batch_size = 1
    partial_pipe = video_input_pipeline(input_name=input_name, batch_size=batch_size,
                                        frames_per_sequence=frames_per_sequence, device=device,
                                        last_sequence_policy='partial', **common_pipeline_params)
    pad_pipe = video_input_pipeline(input_name=input_name, batch_size=batch_size,
                                    frames_per_sequence=frames_per_sequence, device=device,
                                    last_sequence_policy='pad', **common_pipeline_params)

    test_video = files[0]
    num_frames = get_num_frames(test_video)

    partial_pipe.build()
    partial_pipe.feed_input(input_name, np.array([[test_video]]))
    pad_pipe.build()
    pad_pipe.feed_input(input_name, np.array([[test_video]]))

    num_full_sequences = num_frames // frames_per_sequence
    for _ in range(num_full_sequences+1):
        out1 = partial_pipe.run()
        out2 = pad_pipe.run()
        assert np.all(out1[0].as_array() == out2[0].as_array())

    num_frames_in_partial_sequence = num_frames - (num_full_sequences*frames_per_sequence)
    partial_out = partial_pipe.run()
    pad_out = pad_pipe.out()
    #
    # partial_sequence = partial_out[0].as_array()[0]
    # pad_sequence = pad_out[0].as_array()[0]
    #
    # empty_frame = np.zeros()

# def test_mytest():
#     input_name = "VIDEO_INPUT"
#     decoder_pipe = video_decoder_pipeline(input_name=input_name, batch_size=1, num_threads=1,
#                                           device_id=0, exec_pipelined=False, exec_async=False,
#                                           device='cpu')
#     decoder_pipe.build()
#     decoder_pipe.feed_input(input_name, [files[0]])
#     decoder_out = decoder_pipe.run()
#     ref_sequence = decoder_out[0].as_array()[0]
#     num_frames = ref_sequence.shape[0]
#     import ipdb; ipdb.set_trace()
#     pass
