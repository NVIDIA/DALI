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
import shutil
import tempfile
import unittest
import uuid
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
# from the right ones in a small area only: compare frames exactly.
PACKED_BFRAMES_AVI = DALI_EXTRA_PATH + "/db/video/ucf101_test/packed_bframes_test.avi"


def _sequential_gpu_decode(path):
    """Returns all frames of `path`, decoded sequentially by experimental.readers.video."""

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[path], sequence_length=1, name="r"
        )

    p = pipe()
    p.build()
    return [np.array(p.run()[0].as_cpu()[0])[0] for _ in range(p.reader_meta("r")["epoch_size"])]


def _assert_sequence_equals_sequential_decode(case, first, sequence, reference):
    for k, frame in enumerate(sequence):
        if not np.array_equal(frame, reference[first + k]):
            diffs = [np.abs(np.int16(frame) - np.int16(ref)).mean() for ref in reference]
            raise AssertionError(
                f"[{case}] frame {first + k} (frame {k} of the sequence starting at {first}) "
                f"differs from the sequentially decoded one (mean abs diff "
                f"{diffs[first + k]:.2f}); closest sequentially decoded frame: "
                f"{int(np.argmin(diffs))} (mean abs diff {min(diffs):.2f})"
            )


def test_packed_bframes_avi_seek_matches_sequential_decode():
    """MPEG-4 Part 2 in AVI goes through the stateful mpeg4_unpack_bframes bitstream filter,
    which holds back the B-frame of a packed P+B packet until the next packet. The GPU decoder
    used to keep that held-back B-frame across seeks, so after a seek it was decoded in place of
    the keyframe: the pictures no longer matched their timestamps, the requested frames were
    never written and the samples kept stale content (e.g. the frames of an earlier sample)."""
    if not os.path.isfile(PACKED_BFRAMES_AVI):
        raise unittest.SkipTest(
            f"{PACKED_BFRAMES_AVI} not found: DALI_extra is too old to contain the "
            "db/video/ucf101_test fixture"
        )
    reference = _sequential_gpu_decode(PACKED_BFRAMES_AVI)
    assert len(reference) == 91, f"expected 91 frames, got {len(reference)}"
    sequence_length = 4
    starts = [0, 63, 28, 77, 56, 14, 7, 21, 49, 70, 84, 42, 35]
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
        _assert_sequence_equals_sequential_decode(
            "file_list", first, np.array(video.as_cpu()[0]), reference
        )




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
