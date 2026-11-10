# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import atexit
import enum
import functools
import itertools
import random
import shutil
import tempfile
import unittest
import uuid
from test_utils import get_dali_extra_path
from nose2.tools import cartesian_params, params
from nose_utils import assert_raises

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
MULTIPLE_RESOLUTION_ROOT = "/tmp/video_resolution/vp9/"
VFR_VIDEO_FILE = os.path.join(DALI_EXTRA_PATH, "db", "video", "vfr_test.mp4")
CFR_VIDEO_FILE = os.path.join(DALI_EXTRA_PATH, "db", "video", "cfr_test.mp4")

# VP9, 24 fps, 240 frames, timebase 1/12288, first pts 0 (cut with ffmpeg -ss 0 -t 10 by
# qa/TL0_videoreader_test/test.sh). Pinned by test_video_0_fixture_assumptions.
VIDEO_0 = sorted(VIDEO_FILES)[0]

# VP9, 60 fps, 50 frames, first pts 0 (see db/video/cfr/README.txt in DALI_extra).
CFR_VP9_60FPS_FILE = os.path.join(DALI_EXTRA_PATH, "db", "video", "cfr", "test_1_vp9.mp4")

# VP9, full ("pc") color range (see db/video/full_dynamic_range/README.rst in DALI_extra).
FULL_RANGE_VP9_FILE = os.path.join(
    DALI_EXTRA_PATH, "db", "video", "full_dynamic_range", "video_vp9.mp4"
)

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
    # Compare frames. `diff_step` is expressed in the 8-bit [0, 255] intensity scale, so frames
    # must be given in that scale (normalized float frames should be multiplied by 255 first).
    diff_pixels = np.count_nonzero(np.abs(np.float32(frame) - np.float32(ref_frame)) > diff_step)
    total_pixels = frame.size
    # More than threshold of the pixels differ in more than 2 steps
    if diff_pixels / total_pixels > threshold:
        # Save the mismatched frames for inspection
        out_dir = tempfile.mkdtemp(prefix="dali_video_reader_mismatch_")
        frame_u8 = np.uint8(np.clip(np.round(frame), 0, 255))
        ref_frame_u8 = np.uint8(np.clip(np.round(ref_frame), 0, 255))
        suffix = f"{iteration_idx:03d}_{batch_idx:03d}_{frame_idx:03d}.png"
        cv2.imwrite(
            os.path.join(out_dir, "frame_" + suffix), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(out_dir, "ref_frame_" + suffix),
            cv2.cvtColor(ref_frame_u8, cv2.COLOR_RGB2BGR),
        )
        assert False, (
            f"Frame {frame_idx} of sample {batch_idx} in iteration {iteration_idx} differs from "
            f"reference by more than {diff_step} steps in {diff_pixels / total_pixels * 100:.2f}% "
            f"of pixels (threshold: {threshold}). Frames saved to {out_dir}."
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
                    assert np.array_equal(sample_legacy, sample_experimental)
                break
            break
        break


_legacy_gpu_reader = functools.partial(fn.readers.video, device="gpu")


def _experimental_reader(device):
    return functools.partial(fn.experimental.readers.video, device=device)


_file_list_dir = tempfile.mkdtemp(prefix="dali_video_reader_file_lists_")
atexit.register(shutil.rmtree, _file_list_dir, ignore_errors=True)


def _write_file_list(lines):
    """Writes the given file_list entries (one string per line) to a new temporary file and
    returns its path. The files are removed when the test process exits."""
    path = os.path.join(_file_list_dir, f"file_list_{uuid.uuid4().hex}.txt")
    with open(path, "w") as list_file:
        list_file.write("".join(line + "\n" for line in lines))
    return path


def _collect_samples(reader_fn, **kwargs):
    """Runs exactly one epoch of `reader_fn` with batch_size=1 and returns
    (epoch_size, samples), where `samples` is a list of (label, first_frame_num, num_frames)
    tuples in reader order.

    The reader configuration must produce labels (file_list, file_root, or filenames+labels).
    """

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        video, label, frame_num = reader_fn(name="reader", enable_frame_num=True, **kwargs)
        return video, label, frame_num

    p = pipe()
    p.build()
    epoch_size = p.reader_meta("reader")["epoch_size"]
    samples = []
    for _ in range(epoch_size):
        video, label, frame_num = p.run()
        samples.append(
            (
                int(np.array(label.as_cpu()[0]).flatten()[0]),
                int(np.array(frame_num.as_cpu()[0]).flatten()[0]),
                int(video.shape()[0][0]),
            )
        )
    return epoch_size, samples


def _file_list_frame_selection(reader_fn, file_list, **kwargs):
    """Returns (epoch_size, sorted start frame indices) for a single-frame-per-sample reader."""
    epoch_size, samples = _collect_samples(
        reader_fn, file_list=file_list, sequence_length=1, step=1, **kwargs
    )
    return epoch_size, sorted(frame_num for _, frame_num, _ in samples)


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
    lines = []
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
        lines.append(f"{file} {label} {start} {end}")
    list_file = _write_file_list(lines)

    if debug:
        print("File list contents:")
        print("\n".join(lines))

    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        file_list=list_file,
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
        assert np.array_equal(
            fn_arr, expected_idxs
        ), f"Video {i}: frame index mismatch (num_frames={n})"

        if device == "gpu":
            uniform_frames = np.array(video.evaluate().cpu())  # shape (k, H, W, C)
            expected_frames = all_frames[i][expected_idxs]  # shape (k, H, W, C)
            assert np.array_equal(
                uniform_frames, expected_frames
            ), f"Video {i}: pixel mismatch at linspace positions"


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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as list_file:
        list_file.write(f"{video_file} 0 {start_frame} {end_frame}\n")
        list_file.flush()

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

        _, _, frame_num = samples[0]
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
        assert np.array_equal(
            fn_arr, expected_idxs
        ), f"Frame index mismatch (start={start_frame}, end={end_frame})"

        # Scalar mode: enable_frame_num="scalar" should return the first sampled frame index
        # (= start_frame), even with a non-zero ROI offset.
        scalar_reader = ndd.experimental.readers.Video(
            device=device,
            file_list=list_file.name,
            file_list_format="frames",
            sequence_length=sequence_length,
            uniform_sample=True,
            enable_frame_num="scalar",
        )
        scalar_samples = list(scalar_reader.next_epoch())
        assert len(scalar_samples) == 1
        _, _, scalar_fn = scalar_samples[0]
        scalar_val = int(np.array(scalar_fn.evaluate().cpu()).flatten()[0])
        assert (
            scalar_val == start_frame
        ), f"Scalar frame_num should be {start_frame} (first sampled index), got {scalar_val}"


@cartesian_params(devices)
def test_uniform_sample_sequence_length_zero_raises(device):
    """uniform_sample=True with sequence_length=0 should raise an error."""
    video_file = sorted(VIDEO_FILES)[0]
    try:
        reader = ndd.experimental.readers.Video(
            device=device,
            filenames=[video_file],
            sequence_length=0,
            uniform_sample=True,
        )
        reader.get_metadata()  # force backend initialization to trigger validation
        assert False, "Expected an exception for sequence_length=0 with uniform_sample=True"
    except RuntimeError:
        pass  # expected


@cartesian_params(devices)
def test_uniform_sample_stride_step_ignored(device):
    """Passing stride/step with uniform_sample=True should not affect the output."""
    video_file = sorted(VIDEO_FILES)[0]
    sequence_length = 5

    reader_default = ndd.experimental.readers.Video(
        device=device,
        filenames=[video_file],
        sequence_length=sequence_length,
        uniform_sample=True,
        enable_frame_num="sequence",
    )
    # stride and step are explicitly provided but should be ignored.
    reader_with_stride_step = ndd.experimental.readers.Video(
        device=device,
        filenames=[video_file],
        sequence_length=sequence_length,
        uniform_sample=True,
        stride=7,
        step=13,
        enable_frame_num="sequence",
    )

    samples_default = list(reader_default.next_epoch())
    samples_with_stride_step = list(reader_with_stride_step.next_epoch())

    assert len(samples_default) == len(samples_with_stride_step) == 1
    _, fn_default = samples_default[0]
    _, fn_stride_step = samples_with_stride_step[0]
    idxs_default = np.array(fn_default.evaluate().cpu()).flatten()
    idxs_stride_step = np.array(fn_stride_step.evaluate().cpu()).flatten()
    assert np.array_equal(
        idxs_default, idxs_stride_step
    ), "stride/step should be ignored when uniform_sample=True"


@cartesian_params(devices)
def test_video_0_fixture_assumptions(device):
    # Tests that hard-code frame numbers rely on VIDEO_0 being 240 frames at 24 fps. Also checks
    # the other pinned fixtures are present, and that the shared helpers agree with the legacy
    # reader on a trivial selection.
    for path in (CFR_VP9_60FPS_FILE, FULL_RANGE_VP9_FILE, VFR_VIDEO_FILE, CFR_VIDEO_FILE):
        assert os.path.isfile(path), f"Missing test fixture {path}."

    epoch_size, samples = _collect_samples(
        _experimental_reader(device), filenames=[VIDEO_0], labels=[7], sequence_length=1
    )
    assert epoch_size == 240
    assert samples == [(7, i, 1) for i in range(240)]

    list_file = _write_file_list([f"{VIDEO_0} 0 5 15"])
    experimental = _file_list_frame_selection(
        _experimental_reader(device), list_file, file_list_format="frames", pad_mode="none"
    )
    legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file, file_list_frame_num=True)
    assert experimental == legacy, f"experimental: {experimental}, legacy: {legacy}"


# MPEG-4 Part 2 (Xvid) in AVI with packed B-frames, 91 frames showing their frame number on a
# plain background (see db/video/ucf101_test/README.rst in DALI_extra), so wrong frames differ
# from the right ones in a small area only.
PACKED_BFRAMES_AVI = os.path.join(
    DALI_EXTRA_PATH, "db", "video", "ucf101_test", "packed_bframes_test.avi"
)


def _opencv_frames(path):
    """Yields the frames of `path` (RGB) in display order, decoded by OpenCV, which is independent
    of the DALI decoders."""
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


@functools.lru_cache(maxsize=4)
def _display_order_reference(path):
    """Returns the frames of `path` in display order, decoded by OpenCV, to identify DALI's frames
    with _identify_frame: `truth` (frames, pixels, channels) holds them in the `mask` area only,
    where they differ from each other. The colour conversion differs from the one of the DALI GPU
    decoder, so these frames are only meant to tell which frame is which, not to be compared byte
    by byte."""
    if not os.path.isfile(path):
        raise unittest.SkipTest(f"{path} not found: DALI_extra is too old")
    # Two passes keep the memory low: the spread of each pixel, then the pixels that spread
    count, total, total_sq = 0, 0.0, 0.0
    for frame in _opencv_frames(path):
        frame = frame.astype(np.float64)
        count, total, total_sq = count + 1, total + frame, total_sq + frame * frame
    std = np.sqrt(np.maximum(total_sq / count - (total / count) ** 2, 0))
    mask = std.max(-1) > 20
    assert mask.any()
    truth = np.stack([np.int16(frame[mask]) for frame in _opencv_frames(path)])
    return truth, mask


def _identify_frame(frame, truth, mask):
    """Returns the index of the frame in `truth` that `frame` is closest to, in the `mask` area (see
    _display_order_reference)."""
    return int(np.abs(np.int16(frame[mask])[None] - truth).mean((1, 2)).argmin())


def test_packed_bframes_avi_seek_shows_requested_frames():
    """MPEG-4 Part 2 in AVI goes through the stateful mpeg4_unpack_bframes bitstream filter,
    which holds back the B-frame of a packed P+B packet until the next packet. The GPU decoder
    used to keep that held-back B-frame across seeks, so after a seek it was decoded in place of
    the keyframe: the pictures no longer matched their timestamps, the requested frames were
    never written and the samples kept stale content (e.g. the frames of an earlier sample).

    The frames are checked against an independent (OpenCV) decode in display order, rather than
    against the sequential output of the reader under test, so the test also catches frames
    that are returned in decode order."""
    if not os.path.isfile(PACKED_BFRAMES_AVI):
        raise unittest.SkipTest(
            f"{PACKED_BFRAMES_AVI} not found: DALI_extra is too old to contain the "
            "db/video/ucf101_test fixture"
        )
    # The pixels that differ between frames: the area where the frame number is drawn.
    truth, mask = _display_order_reference(PACKED_BFRAMES_AVI)
    # The last of the 91 frames is a not-coded VOP, which repeats frame 89: OpenCV returns 90.
    assert len(truth) >= 90, f"expected at least 90 frames, got {len(truth)}"

    sequence_length = 4
    starts = [0, 63, 28, 77, 56, 14, 7, 21, 49, 70, 84, 42, 35]
    assert max(starts) + sequence_length <= len(truth)
    list_file = _write_file_list(
        [f"{PACKED_BFRAMES_AVI} 0 {s} {s + sequence_length}" for s in starts]
    )

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        video, _, frame_num = fn.experimental.readers.video(
            device="gpu",
            file_list=list_file,
            file_list_format="frames",
            sequence_length=sequence_length,
            enable_frame_num=True,
        )
        return video, frame_num

    p = pipe()
    p.build()
    for s in starts:
        video, frame_num = p.run()
        first = int(np.array(frame_num.as_cpu()[0]).flatten()[0])
        assert first == s, f"expected the sequence starting at frame {s}, got {first}"
        for k, frame in enumerate(np.array(video.as_cpu()[0])):
            shown = _identify_frame(frame, truth, mask)
            assert shown == first + k, (
                f"frame {first + k} (frame {k} of the sequence starting at {first}) "
                f"shows frame {shown}"
            )


# AVI files with B-frames: AVI stores no presentation timestamps, only the order of the packets,
# which is the decoding order. DivX packed B-frames (MPEG-4 Part 2, 91 frames) and H.264 with
# pyramid B-frames (180 frames).
AVI_WITH_BFRAMES = (
    PACKED_BFRAMES_AVI,
    os.path.join(DALI_EXTRA_PATH, "db", "video", "containers", "avi", "cfr.avi"),
)


def _check_frames_shown(path, frames, video, truth, mask):
    """Checks that `video` holds the frames `frames` of `path`, in display order."""
    for k, frame in zip(frames, video, strict=True):
        # The packed fixture has a 91st frame, a not-coded VOP, which repeats the previous one and
        # which OpenCV does not return
        expected = min(k, len(truth) - 1)
        shown = _identify_frame(frame, truth, mask)
        assert shown == expected, f"{os.path.basename(path)}: frame {k} shows frame {shown}"


@params(*AVI_WITH_BFRAMES)
def test_avi_with_bframes_read_in_display_order(path):
    """The GPU reader returned the frames of an AVI file with B-frames in decoding order: it took
    their order from the timestamps of the packets, which AVI gives in decoding order. The frames
    are read in a row, and then with seeks backward and forward."""
    truth, mask = _display_order_reference(path)
    sequence_length = 10
    num_frames = 91 if path == PACKED_BFRAMES_AVI else len(truth)
    starts = list(range(0, num_frames - sequence_length + 1, sequence_length))
    starts += [num_frames - sequence_length, 3, 45, 12, num_frames - sequence_length - 1]
    list_file = _write_file_list([f"{path} 0 {s} {s + sequence_length}" for s in starts])

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        video, _, frame_num = fn.experimental.readers.video(
            device="gpu",
            file_list=list_file,
            file_list_format="frames",
            sequence_length=sequence_length,
            enable_frame_num=True,
        )
        return video, frame_num

    p = pipe()
    p.build()
    for s in starts:
        video, frame_num = p.run()
        first = int(np.array(frame_num.as_cpu()[0]).flatten()[0])
        assert first == s, f"expected the sequence starting at frame {s}, got {first}"
        video = np.array(video.as_cpu()[0])
        _check_frames_shown(path, range(s, s + sequence_length), video, truth, mask)
        if path == PACKED_BFRAMES_AVI and s + sequence_length == num_frames:
            # The not-coded VOP repeats the frame before it exactly
            np.testing.assert_array_equal(video[-1], video[-2])


@params(*itertools.product(AVI_WITH_BFRAMES, (True, False)))
def test_avi_with_bframes_decoded_in_display_order(path, build_index):
    """As test_avi_with_bframes_read_in_display_order, for the video decoder: with an index, and
    without one (the frames in the order that the decoder outputs them)."""
    truth, mask = _display_order_reference(path)
    # Without an index, the frames are the pictures: the not-coded VOP is not one
    last = 90 if path == PACKED_BFRAMES_AVI and build_index else len(truth) - 1
    frames = [0, 1, 2, 3, 4, 5, 6, 7, 30, 31, 32, 33, 15, 16, last - 2, last - 1, last, 8, 9]

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        encoded, _ = fn.readers.file(files=[path])
        return fn.experimental.decoders.video(
            encoded, device="mixed", frames=frames, build_index=build_index
        )

    p = pipe()
    p.build()
    (video,) = p.run()
    _check_frames_shown(path, frames, np.array(video.as_cpu()[0]), truth, mask)


# MPEG-4 Part 2 in AVI with B-frames and open GOPs (FFmpeg's encoder): 90 frames of 160x128, a
# keyframe about every 15, each frame showing its number as the one bright cell of a 10x9 grid of
# 16x14 cells. qa/TL0_videoreader_test/test.sh makes it.
MPEG4_OPEN_GOP_AVI = "/tmp/avi_video_files/mpeg4_open_gop.avi"


def _frame_number(frame):
    """The number that a frame of MPEG4_OPEN_GOP_AVI shows: the index of its bright cell."""
    cells = np.float32(frame).mean(-1)[:126, :160].reshape(9, 14, 10, 16).mean((1, 3))
    return int(cells.argmax())


@params("reader", "decoder")
def test_avi_open_gop_seeks_show_requested_frames(op):
    """The keyframes of an open GOP have leading pictures: shown before the keyframe, but decoded
    after it from the pictures before it. In AVI, whose timestamps follow the decoding order, they
    take the frames right after the keyframe, so a seek to one of them has to start at the keyframe
    before. Every frame is read once, in a shuffled order, so that most reads seek, also to the
    leading pictures and to the keyframes."""
    if not os.path.isfile(MPEG4_OPEN_GOP_AVI):
        raise unittest.SkipTest(f"{MPEG4_OPEN_GOP_AVI} not found: made by the qa test script")
    num_frames = 90
    order = list(range(num_frames))
    random.Random(42).shuffle(order)

    if op == "reader":
        sequence_length = 3
        starts = [s for s in order if s + sequence_length <= num_frames]
        list_file = _write_file_list(
            [f"{MPEG4_OPEN_GOP_AVI} 0 {s} {s + sequence_length}" for s in starts]
        )

        @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
        def pipe():
            video, _ = fn.experimental.readers.video(
                device="gpu",
                file_list=list_file,
                file_list_format="frames",
                sequence_length=sequence_length,
            )
            return video

        p = pipe()
        p.build()
        reads = [(s, np.array(p.run()[0].as_cpu()[0])) for s in starts]
    else:
        # One frame per iteration: the frames of one sample are decoded in order, without seeks
        @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
        def pipe():
            encoded, _ = fn.readers.file(files=[MPEG4_OPEN_GOP_AVI])
            frame = fn.external_source(
                source=lambda info: np.array([order[info.iteration]], dtype=np.int32),
                batch=False,
            )
            return fn.experimental.decoders.video(encoded, device="mixed", frames=frame)

        p = pipe()
        p.build()
        reads = [(f, np.array(p.run()[0].as_cpu()[0])) for f in order]

    for first, frames in reads:
        for k, frame in enumerate(frames):
            shown = _frame_number(frame)
            assert shown == first + k, f"frame {first + k} shows frame {shown}"


class _Reader(enum.Enum):
    LEGACY = enum.auto()
    EXPERIMENTAL_CPU = enum.auto()
    EXPERIMENTAL_GPU = enum.auto()

    def make(self):
        if self is _Reader.LEGACY:
            return _legacy_gpu_reader
        return _experimental_reader("cpu" if self is _Reader.EXPERIMENTAL_CPU else "gpu")


class _PathKind(enum.Enum):
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()


@cartesian_params(list(_Reader), list(_PathKind))
def test_filename_with_colon_can_be_opened(reader, path_kind):
    """A file name such as `clip:01.mp4` must not be parsed as a URL with protocol `clip`: the
    file opens and all 50 frames are indexed, like any other file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        abs_name = os.path.join(tmp_dir, "clip:01.mp4")
        shutil.copy(CFR_VP9_60FPS_FILE, abs_name)
        name = abs_name if path_kind is _PathKind.ABSOLUTE else "clip:01.mp4"
        reader_fn = reader.make()

        @pipeline_def(batch_size=1, num_threads=2, device_id=0)
        def pipe():
            return reader_fn(filenames=[name], sequence_length=1, name="r")

        prev_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            try:
                p = pipe()
                p.build()
            except RuntimeError as e:
                raise AssertionError(
                    f"[{reader.name}, {path_kind.name} path] could not open {name!r}: {e}."
                )
            epoch_size = p.reader_meta("r")["epoch_size"]
            assert epoch_size == 50, (
                f"[{reader.name}, {path_kind.name} path] indexed {epoch_size} frames of {name!r}, "
                f"expected 50."
            )
            p.run()
        finally:
            os.chdir(prev_cwd)


@cartesian_params(devices)
def test_file_url_can_be_opened(device):
    """A `file://` URL names a local file and must open like a plain path (experimental reader
    only: the legacy reader does not accept URLs)."""
    url = "file://" + os.path.abspath(CFR_VP9_60FPS_FILE)

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        return _experimental_reader(device)(filenames=[url], sequence_length=1, name="r")

    p = pipe()
    p.build()
    assert p.reader_meta("r")["epoch_size"] == 50
    p.run()


@cartesian_params(devices)
def test_empty_labels_means_sequential_labels(device):
    files = sorted(VIDEO_FILES)[:3]
    # step=1000 > 240 frames: exactly one sample per file.
    kwargs = dict(filenames=files, labels=[], sequence_length=10, step=1000)
    experimental = _collect_samples(_experimental_reader(device), **kwargs)
    legacy = _collect_samples(_legacy_gpu_reader, **kwargs)
    assert (
        experimental == legacy == (3, [(0, 0, 10), (1, 0, 10), (2, 0, 10)])
    ), f"experimental: {experimental}, legacy: {legacy}"


@cartesian_params(devices)
def test_labels_not_passed_means_no_labels_output(device):
    # Regression guard for the other side of the distinction: no `labels` -> no labels output.
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        outs = fn.experimental.readers.video(device=device, filenames=[VIDEO_0], sequence_length=3)
        assert isinstance(outs, DataNode), f"Expected only the video output, got {len(outs)}"
        return outs

    p = pipe()
    p.build()
    p.run()


@cartesian_params(devices)
def test_empty_string_sources_are_treated_as_not_provided(device):
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def filenames_pipe():
        outs = fn.experimental.readers.video(
            device=device, filenames=[VIDEO_0], file_root="", file_list="", sequence_length=3
        )
        # No labels are requested, so only the video output may be declared.
        assert isinstance(outs, DataNode), f"Expected only the video output, got {len(outs)}"
        return outs

    p = filenames_pipe()
    p.build()
    (video,) = p.run()
    assert video.shape()[0][0] == 3

    list_file = _write_file_list([f"{VIDEO_0} 7 0 30"])

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def file_list_pipe():
        video, label = fn.experimental.readers.video(
            device=device,
            file_list=list_file,
            filenames=[],
            file_root="",
            file_list_format="frames",
            sequence_length=3,
        )
        return video, label

    p = file_list_pipe()
    p.build()
    _, label = p.run()
    assert int(np.array(label.as_cpu()[0]).flatten()[0]) == 7


@cartesian_params(devices)
def test_empty_companions_of_file_root_are_treated_as_not_provided(device):
    def epoch_size(**kwargs):
        @pipeline_def(batch_size=1, num_threads=2, device_id=0)
        def pipe():
            video, label = fn.experimental.readers.video(
                device=device, sequence_length=3, name="reader", **kwargs
            )
            return video, label

        p = pipe()
        p.build()
        return p.reader_meta("reader")["epoch_size"]

    reference = epoch_size(file_root=MULTIPLE_RESOLUTION_ROOT)
    with_empty = epoch_size(file_root=MULTIPLE_RESOLUTION_ROOT, file_list="", filenames=[])
    assert with_empty == reference


@cartesian_params(devices)
def test_all_sources_empty_raise(device):
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device=device, filenames=[], file_root="", file_list="", sequence_length=3
        )

    with assert_raises(RuntimeError, glob="*Exactly one of*"):
        p = pipe()
        p.build()


@cartesian_params(devices)
def test_two_non_empty_sources_raise(device):
    list_file = _write_file_list([f"{VIDEO_0} 0 0 30"])

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        # Both filenames and file_list are provided, which should raise an error.
        # The output function sees file_list is non-empty, so it declares 2 outputs.
        # We unpack both to avoid Python validation errors, letting the C++ error occur.
        video, label = fn.experimental.readers.video(
            device=device, filenames=[VIDEO_0], file_list=list_file, sequence_length=3
        )
        return video, label

    with assert_raises(RuntimeError, glob="*Exactly one of*"):
        p = pipe()
        p.build()  # Error occurs in the loader constructor, triggered by build()
