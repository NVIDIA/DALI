# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import os
import itertools
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nose2.tools import params
from nose_utils import assert_raises
from nvidia.dali import pipeline_def
from test_utils import get_dali_extra_path, to_array

test_data_root = get_dali_extra_path()
filenames = glob.glob(f"{test_data_root}/db/video/[cv]fr/*.mp4")
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: "hevc" not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator
filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
filenames = filter(lambda filename: "av1" not in filename, filenames)
files = [np.fromfile(filename, dtype=np.uint8) for filename in filenames]

batch_size_values = [1, 3, 100]
frames_per_sequence_values = [1, 7, 100]
device_values = ["cpu", "mixed"]


def params_generator():
    """
    Generates parameters for test.
    To be used in the @params decorator:
    @params(*list(params_generator())

    The pattern of generating the parameters:
    1. Generating a set of all permutations of `batch_size`, `frames_per_sequence` and `device`.
    2. Assigning the test file in a round-robin fashion to every permutation of parameters.
    """
    test_params = (
        (dev, fps, bs)
        for dev in device_values
        for fps in frames_per_sequence_values
        for bs in batch_size_values
    )
    num_test_files = len(files)
    file_idx = 0
    for tp in test_params:
        yield tp + (files[file_idx],)
        file_idx = (file_idx + 1) % num_test_files


@pipeline_def
def video_decoder_pipeline(input_name, device="cpu"):
    data = fn.external_source(name=input_name, dtype=types.UINT8, ndim=1)
    vid = fn.experimental.decoders.video(data, device=device)
    return vid


@pipeline_def
def video_input_pipeline(input_name, sequence_length, last_sequence_policy="partial", device="cpu"):
    vid = fn.experimental.inputs.video(
        name=input_name,
        device=device,
        blocking=False,
        sequence_length=sequence_length,
        last_sequence_policy=last_sequence_policy,
    )
    return vid


# Parameters common for the DALI pipelines used throughout this test.
common_pipeline_params = {
    "num_threads": 1,
    "device_id": 0,
    "exec_pipelined": False,
    "exec_async": False,
    "prefetch_queue_depth": 1,
}


def get_num_frames(encoded_video):
    input_name = "VIDEO_INPUT"
    decoder_pipe = video_decoder_pipeline(
        input_name=input_name, batch_size=1, device="cpu", **common_pipeline_params
    )
    decoder_pipe.feed_input(input_name, [encoded_video])
    decoder_out = decoder_pipe.run()
    return decoder_out[0].as_array()[0].shape[0]


def get_batch_outline(num_frames, frames_per_sequence, batch_size):
    num_iterations = num_frames // (frames_per_sequence * batch_size)
    remaining_frames = num_frames - num_iterations * frames_per_sequence * batch_size
    num_full_sequences = remaining_frames // frames_per_sequence
    num_frames_in_partial_sequence = remaining_frames - num_full_sequences * frames_per_sequence
    return num_iterations, num_full_sequences, num_frames_in_partial_sequence


def portion_out_reference_sequence(decoder_pipe_out, frames_per_sequence, batch_size):
    """
    A generator, that takes the output from VideoDecoder DALI pipeline. Then, based of the
    provided parameters, it serves sequences one-by-one, which are supposed to be returned
    by VideoInput operator.
    """
    ref_sequence = to_array(decoder_pipe_out[0])[0]
    num_frames = ref_sequence.shape[0]
    n_batches = num_frames // (batch_size * frames_per_sequence)
    ref_sequence = ref_sequence[: n_batches * frames_per_sequence * batch_size]
    sh = ref_sequence.shape
    ref_sequence = ref_sequence.reshape(
        n_batches, batch_size, frames_per_sequence, sh[1], sh[2], sh[3]
    )
    for rs in ref_sequence:
        yield rs


@params(*list(params_generator()))
def test_video_input_compare_with_video_decoder(device, frames_per_sequence, batch_size, test_file):
    """
    Compares the VideoInput with the VideoDecoder.
    """
    input_name = "VIDEO_INPUT"

    decoder_pipe = video_decoder_pipeline(
        input_name=input_name, batch_size=1, device=device, **common_pipeline_params
    )
    input_pipe = video_input_pipeline(
        input_name=input_name,
        batch_size=batch_size,
        sequence_length=frames_per_sequence,
        device=device,
        **common_pipeline_params,
    )

    decoder_pipe.feed_input(input_name, [test_file])
    decoder_out = decoder_pipe.run()

    input_pipe.feed_input(input_name, np.array([[test_file]]))

    for ref_seq in portion_out_reference_sequence(decoder_out, frames_per_sequence, batch_size):
        input_out = input_pipe.run()
        test_seq = to_array(input_out[0])
        assert np.all(ref_seq == test_seq)


@params(*list(params_generator()))
def test_video_input_partial_vs_pad(device, frames_per_sequence, batch_size, test_video):
    input_name = "VIDEO_INPUT"
    partial_pipe = video_input_pipeline(
        input_name=input_name,
        batch_size=batch_size,
        sequence_length=frames_per_sequence,
        device=device,
        last_sequence_policy="partial",
        **common_pipeline_params,
    )
    pad_pipe = video_input_pipeline(
        input_name=input_name,
        batch_size=batch_size,
        sequence_length=frames_per_sequence,
        device=device,
        last_sequence_policy="pad",
        **common_pipeline_params,
    )

    num_frames = get_num_frames(test_video)

    partial_pipe.feed_input(input_name, np.array([[test_video]]))
    pad_pipe.feed_input(input_name, np.array([[test_video]]))

    num_iterations, num_full_sequences, num_frames_in_partial_sequence = get_batch_outline(
        num_frames, frames_per_sequence, batch_size
    )

    # First, check all the full batches with full sequences
    for _ in range(num_iterations):
        out1 = partial_pipe.run()
        out2 = pad_pipe.run()
        np.testing.assert_array_equal(to_array(out1[0]), to_array(out2[0]))

    if num_frames - num_iterations * frames_per_sequence * batch_size == 0:
        # Frames have been split equally across batches.
        return

    # Now check the full sequences in the last batch
    partial_out = partial_pipe.run()[0]
    pad_out = pad_pipe.run()[0]
    for i in range(num_full_sequences):
        np.testing.assert_array_equal(to_array(partial_out[i]), to_array(pad_out[i]))

    # And lastly, the actual check PARTIAL vs PAD -
    # the last sequence in the last batch, which might be partial (or padded).
    if num_frames_in_partial_sequence == 0:
        return
    last_partial_sequence = to_array(partial_out[num_full_sequences])
    last_pad_sequence = to_array(pad_out[num_full_sequences])
    for i in range(num_frames_in_partial_sequence):
        # The frames that are in both - partial and padded sequences.
        np.testing.assert_array_equal(last_partial_sequence[i], last_pad_sequence[i])
    frame_shape = last_pad_sequence[0].shape
    empty_frame = np.zeros(frame_shape, dtype=np.uint8)
    for i in range(num_frames_in_partial_sequence, frames_per_sequence):
        # The frames that are only in padded sequence.
        np.testing.assert_array_equal(last_pad_sequence[i], empty_frame)


@params(*itertools.product(device_values, (1, 4)))
def test_video_input_input_queue(device, n_test_files):
    """
    Checks the input queue on `fn.inputs.video` operator.
    """
    input_name = "VIDEO_INPUT"
    batch_size = 3
    frames_per_sequence = 4

    input_pipe = video_input_pipeline(
        input_name=input_name,
        batch_size=batch_size,
        sequence_length=frames_per_sequence,
        device=device,
        **common_pipeline_params,
    )

    for i in range(n_test_files):
        input_pipe.feed_input(input_name, np.array([[files[i]]]))

    n_runs = 0
    for i in range(n_test_files):
        num_frames = get_num_frames(files[i])
        ni, nfs, nfips = get_batch_outline(num_frames, frames_per_sequence, batch_size)
        n_runs += ni + (1 if nfs + nfips > 0 else 0)

    for _ in range(n_runs):
        input_pipe.run()
    # If exception has not been thrown, the test pass.

    with assert_raises(
        RuntimeError,
        glob="No data was provided to the InputOperator. Make sure to feed it properly.",
    ):
        input_pipe.run()


@params(*device_values)
def test_video_input_audio_stream(device):
    """
    Checks if video decoding when audio stream is present
    """
    input_name = "VIDEO_INPUT"

    input_pipe = video_input_pipeline(
        input_name=input_name,
        batch_size=3,
        sequence_length=4,
        device=device,
        **common_pipeline_params,
    )

    filename = os.path.join(test_data_root, "db", "video", "sintel", "sintel_trailer-720p.mp4")
    test_file = np.fromfile(filename, dtype=np.uint8)
    input_pipe.feed_input(input_name, np.array([[test_file]]))

    input_pipe.run()
