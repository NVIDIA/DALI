# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.data_node import DataNode
import nvidia.dali.experimental.dynamic as ndd
import numpy as np
import os
import cv2
import tempfile
from test_utils import get_dali_extra_path
from nose2.tools import cartesian_params

np.random.seed(42)
debug = False  # Set to True to print file_list contents and other debug information

DALI_EXTRA_PATH = get_dali_extra_path()

VIDEO_DIRECTORY = "/tmp/video_files"
PLENTY_VIDEO_DIRECTORY = "/tmp/many_video_files"
VIDEO_FILES = os.listdir(VIDEO_DIRECTORY)
PLENTY_VIDEO_FILES = os.listdir(PLENTY_VIDEO_DIRECTORY)
VIDEO_FILES = [VIDEO_DIRECTORY + "/" + f for f in VIDEO_FILES]
PLENTY_VIDEO_FILES = [PLENTY_VIDEO_DIRECTORY + "/" + f for f in PLENTY_VIDEO_FILES]
FILE_LIST = "/tmp/file_list.txt"
MULTIPLE_RESOLUTION_ROOT = "/tmp/video_resolution/"

devices = ["cpu", "gpu"]
sequence_lengths = [3]
batch_sizes = [1, 10]
file_list_formats = ["frames", "timestamps"]
file_list_roundings = ["start_down_end_up", "start_up_end_down"]
pad_modes = ["none", "constant", "edge", "reflect_1001", "reflect_101"]
pad_modes_supported_by_legacy_reader = ["none", "constant"]
image_type_supported_by_legacy_reader = [types.RGB]  # TODO(janton): Add types.YCbCr


def compare_frames(
    frame, ref_frame, iteration_idx, batch_idx, frame_idx, diff_step=2, threshold=0.03
):
    # Compare frames
    diff_pixels = np.count_nonzero(np.abs(np.float32(frame) - np.float32(ref_frame)) > diff_step)
    total_pixels = frame.size
    # More than threshold of the pixels differ in more than 2 steps
    if diff_pixels / total_pixels > threshold:
        # Save the mismatched frames for inspection
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ref_frame_bgr = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)

        output_path = f"frame_{iteration_idx:03d}_{batch_idx:03d}_{frame_idx:03d}.png"
        ref_output_path = f"ref_frame_{iteration_idx:03d}_{batch_idx:03d}_{frame_idx:03d}.png"

        cv2.imwrite(output_path, frame_bgr)
        cv2.imwrite(ref_output_path, ref_frame_bgr)
        assert False, (
            f"Frame {frame_idx+1} differs from reference by more than {diff_step} steps in "
            + f"{diff_pixels/total_pixels*100}% of pixels (threshold: {threshold}). "
            + f"Expected {ref_frame_bgr} but got {frame_bgr}"
        )


def compare_experimental_to_legacy_reader(device, batch_size, **kwargs):
    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1)
    def video_reader_pipeline():
        kwargs_legacy = dict(kwargs)  # Make a copy to avoid modifying the original
        if "file_list_format" in kwargs:
            file_list_format = kwargs["file_list_format"]
            del kwargs_legacy["file_list_format"]
            if file_list_format == "frames":
                kwargs_legacy["file_list_frame_num"] = True
            elif file_list_format == "timestamps":
                kwargs_legacy["file_list_frame_num"] = False
            file_list_rounding = kwargs.get("file_list_rounding", "start_down_end_up")
            del kwargs_legacy["file_list_rounding"]
            kwargs_legacy["file_list_include_preceding_frame"] = (
                file_list_rounding == "start_down_end_up"
            )
        if "pad_mode" in kwargs:
            pad_mode = kwargs["pad_mode"]
            del kwargs_legacy["pad_mode"]
            if pad_mode == "constant":
                kwargs_legacy["pad_sequences"] = True
            elif pad_mode == "none":
                kwargs_legacy["pad_sequences"] = False
            else:
                raise ValueError(f"Unsupported pad_mode {pad_mode} in legacy reader")

        outs0 = fn.readers.video(
            device="gpu",
            name="legacy_reader",
            **kwargs_legacy,
        )
        if isinstance(outs0, DataNode):
            outs0 = (outs0,)

        outs1 = fn.experimental.readers.video(
            device=device,
            name="experimental_reader",
            **kwargs,
        )
        if isinstance(outs1, DataNode):
            outs1 = (outs1,)

        return tuple(list(outs0) + list(outs1))

    pipe = video_reader_pipeline()
    pipe.build()
    legacy_epoch_size = pipe.reader_meta("legacy_reader")["epoch_size"]
    experimental_epoch_size = pipe.reader_meta("experimental_reader")["epoch_size"]
    # The readers calculate the number of frames in the epoch differently,
    # so we need to take the minimum of the two.
    epoch_size = min(legacy_epoch_size, experimental_epoch_size)
    for i in range(epoch_size):
        outs = pipe.run()
        n = len(outs)
        assert n % 2 == 0
        outs_legacy = outs[: n // 2]
        outs_experimental = outs[n // 2 :]
        assert len(outs_legacy) == len(outs_experimental)
        for _, (out_legacy, out_experimental) in enumerate(zip(outs_legacy, outs_experimental)):
            for j, (sample_legacy, sample_experimental) in enumerate(
                zip(out_legacy, out_experimental)
            ):
                sample_legacy = np.array(sample_legacy.as_cpu())
                sample_experimental = np.array(sample_experimental.as_cpu())
                num_frames = sample_legacy.shape[0]
                assert (
                    num_frames == sample_experimental.shape[0]
                ), f"Number of frames mismatch: {num_frames} != {sample_experimental.shape[0]}"
                if i == 0:
                    for k in range(num_frames):
                        compare_frames(sample_experimental[k], sample_legacy[k], i, j, k)
                else:
                    np.testing.assert_array_equal(sample_legacy, sample_experimental)
                break
            break
        break


@cartesian_params(
    devices,
    batch_sizes,
    sequence_lengths,
    pad_modes_supported_by_legacy_reader,
    image_type_supported_by_legacy_reader,
)
def test_compare_experimental_to_legacy_reader_filenames(
    device, batch_size, sequence_length, pad_mode, image_type
):
    labels = [np.random.randint(0, 100) for _ in range(len(VIDEO_FILES))]
    files = VIDEO_FILES
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        filenames=files,
        sequence_length=sequence_length,
        enable_timestamps=True,
        enable_frame_num=True,
        labels=labels,
        pad_mode=pad_mode,
        image_type=image_type,
    )


@cartesian_params(
    devices,
    batch_sizes,
    sequence_lengths,
    file_list_formats,
    file_list_roundings,
    pad_modes_supported_by_legacy_reader,
    image_type_supported_by_legacy_reader,
)
def test_compare_experimental_to_legacy_reader_file_list(
    device, batch_size, sequence_length, file_list_format, file_list_rounding, pad_mode, image_type
):
    files = VIDEO_FILES
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    for i, file in enumerate(files):
        label = np.random.randint(0, 20)
        start = end = 0
        while start >= end:
            if file_list_format == "frames":
                start = np.random.randint(0, 20)
                end = np.random.randint(0, 20)
            else:
                start = np.random.random() * 0.4  # Range [0, 0.4)
                end = 0.6 + np.random.random() * 0.4  # Range [0.6, 1.0)
        list_file.write(f"{file} {label} {start} {end}\n")
    list_file.close()

    if debug:
        print("File list contents:")
        with open(list_file.name, "r") as f:
            print(f.read())

    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        file_list=list_file.name,
        sequence_length=sequence_length,
        enable_timestamps=True,
        enable_frame_num=True,
        file_list_format=file_list_format,
        file_list_rounding=file_list_rounding,
        pad_mode=pad_mode,
        image_type=image_type,
    )


@cartesian_params(
    devices,
    batch_sizes,
    sequence_lengths,
    pad_modes_supported_by_legacy_reader,
    image_type_supported_by_legacy_reader,
)
def test_compare_experimental_to_legacy_reader_file_root(
    device, batch_size, sequence_length, pad_mode, image_type
):
    if debug:
        print("MULTIPLE_RESOLUTION_ROOT contents:")
        for root, dirs, files in os.walk(MULTIPLE_RESOLUTION_ROOT):
            for dir in dirs:
                print(f"Label={dir}, files=[{', '.join(os.listdir(os.path.join(root, dir)))}]")
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        file_root=MULTIPLE_RESOLUTION_ROOT,
        sequence_length=sequence_length,
        pad_mode=pad_mode,
        image_type=image_type,
    )


@cartesian_params(
    devices,
    [5, 1],  # sequence lengths including edge case N=1
)
def test_uniform_sample(device, sequence_length):
    video_files_sorted = sorted(VIDEO_FILES)
    num_video_files = len(video_files_sorted)

    # Get per-video frame counts and (GPU only) decode all frames for pixel comparison.
    # Iterating seq_len=1 gives both in one pass; CPU only needs the count via get_metadata().
    # GPU only: NVDEC produces identical pixels whether seeking or decoding sequentially.
    # CPU (libavcodec) can produce minor seek-induced differences for H.264 B-frames.
    frame_counts = []
    all_frames = []  # populated on GPU only
    for video_file in video_files_sorted:
        reader = ndd.experimental.readers.Video(
            device=device, filenames=[video_file], sequence_length=1, stride=1, step=1
        )
        if device == "gpu":
            decoded = [np.array(f.evaluate().cpu())[0] for (f,) in reader.next_epoch()]
            all_frames.append(np.stack(decoded))  # shape (N, H, W, C)
            frame_counts.append(len(decoded))
        else:
            frame_counts.append(reader.get_metadata()["epoch_size"])

    # Run uniform reader, verify one sample per video, check frame indices and pixels.
    uniform_reader = ndd.experimental.readers.Video(
        device=device,
        filenames=video_files_sorted,
        sequence_length=sequence_length,
        uniform_sample=True,
        enable_frame_num="sequence",
    )
    samples = list(uniform_reader.next_epoch())
    assert (
        len(samples) == num_video_files
    ), f"Expected {num_video_files} samples (one per video), got {len(samples)}"

    for i, (video, frame_num) in enumerate(samples):
        n = frame_counts[i]
        fn_arr = np.array(frame_num.evaluate().cpu()).flatten()
        assert (
            len(fn_arr) == sequence_length
        ), f"Video {i}: expected {sequence_length} frame indices, got {len(fn_arr)}"
        assert fn_arr[0] == 0, f"Video {i}: first frame index should be 0, got {fn_arr[0]}"
        if sequence_length > 1:
            assert (
                fn_arr[-1] == n - 1
            ), f"Video {i}: last frame index should be {n - 1}, got {fn_arr[-1]}"
        # Use floor(x + 0.5) to match C++ std::round (rounds half away from zero).
        expected_idxs = np.floor(np.linspace(0, n - 1, sequence_length) + 0.5).astype(np.int32)
        np.testing.assert_array_equal(
            fn_arr, expected_idxs, err_msg=f"Video {i}: frame index mismatch (num_frames={n})"
        )
        if device == "gpu":
            uniform_frames = np.array(video.evaluate().cpu())  # shape (k, H, W, C)
            expected_frames = all_frames[i][expected_idxs]  # shape (k, H, W, C)
            np.testing.assert_array_equal(
                uniform_frames,
                expected_frames,
                err_msg=f"Video {i}: pixel mismatch at linspace positions",
            )


@cartesian_params(
    devices,
    [5, 1],  # sequence lengths including edge case N=1
)
def test_uniform_sample_file_list_roi(device, sequence_length):
    """Verify uniform_sample with file_list ROI (non-zero start_frame)."""
    video_file = sorted(VIDEO_FILES)[0]

    # Get total frame count.
    reader = ndd.experimental.readers.Video(
        device=device, filenames=[video_file], sequence_length=1, stride=1, step=1
    )
    total_frames = reader.get_metadata()["epoch_size"]

    # Define a ROI that excludes the first and last few frames.
    start_frame = max(1, total_frames // 5)
    end_frame = min(total_frames - 1, total_frames * 4 // 5)
    roi_frames = end_frame - start_frame
    assert roi_frames >= sequence_length, "ROI too small for this test"

    # Write a file_list with the ROI.
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    list_file.write(f"{video_file} 0 {start_frame} {end_frame}\n")
    list_file.close()

    uniform_reader = ndd.experimental.readers.Video(
        device=device,
        file_list=list_file.name,
        file_list_format="frames",
        sequence_length=sequence_length,
        uniform_sample=True,
        enable_frame_num="sequence",
    )
    samples = list(uniform_reader.next_epoch())
    assert len(samples) == 1, f"Expected 1 sample (one per video), got {len(samples)}"

    video, frame_num = samples[0]
    fn_arr = np.array(frame_num.evaluate().cpu()).flatten()
    assert len(fn_arr) == sequence_length

    # Frame indices must be absolute (offset from start_frame, not zero).
    assert fn_arr[0] == start_frame, f"First index should be {start_frame}, got {fn_arr[0]}"
    if sequence_length > 1:
        assert (
            fn_arr[-1] == end_frame - 1
        ), f"Last index should be {end_frame - 1}, got {fn_arr[-1]}"

    expected_idxs = start_frame + np.floor(
        np.linspace(0, roi_frames - 1, sequence_length) + 0.5
    ).astype(np.int32)
    np.testing.assert_array_equal(
        fn_arr,
        expected_idxs,
        err_msg=f"Frame index mismatch (start={start_frame}, end={end_frame})",
    )
