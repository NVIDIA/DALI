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
import atexit
import functools
import shutil
import subprocess
import sys
import tempfile
import uuid
from test_utils import get_dali_extra_path
from nose2.tools import cartesian_params, params
from nose_utils import SkipTest, assert_raises, assert_warns

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
VFR_VIDEO_FILE = DALI_EXTRA_PATH + "/db/video/vfr_test.mp4"
CFR_VIDEO_FILE = DALI_EXTRA_PATH + "/db/video/cfr_test.mp4"

# VP9, 24 fps, 240 frames, timebase 1/12288, first pts 0 (cut with ffmpeg -ss 0 -t 10 by
# qa/TL0_videoreader_test/test.sh). Pinned by test_video_0_fixture_assumptions.
VIDEO_0 = sorted(VIDEO_FILES)[0]

# VP9, 60 fps, 50 frames, first pts 0 (see db/video/cfr/README.txt in DALI_extra).
CFR_VP9_60FPS_FILE = DALI_EXTRA_PATH + "/db/video/cfr/test_1_vp9.mp4"

devices = ["cpu", "gpu"]
sequence_lengths = [3]
batch_sizes = [1, 10]
file_list_formats = ["frames", "timestamps"]
file_list_roundings = ["start_down_end_up", "start_up_end_down"]
pad_modes = ["none", "constant", "edge", "reflect_1001", "reflect_101"]
pad_modes_supported_by_legacy_reader = ["none", "constant"]
image_type_supported_by_legacy_reader = [types.RGB, types.YCbCr]


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
            f"of pixels (threshold: {threshold}). Frames saved to {out_dir}"
        )


def _legacy_reader_kwargs(kwargs):
    """Translates experimental.readers.video arguments to legacy readers.video ones."""
    kwargs_legacy = dict(kwargs)
    if "file_list_format" in kwargs_legacy:
        kwargs_legacy["file_list_frame_num"] = kwargs_legacy.pop("file_list_format") == "frames"
    if "file_list_rounding" in kwargs_legacy:
        rounding = kwargs_legacy.pop("file_list_rounding")
        if rounding not in ("start_up_end_down", "start_down_end_up"):
            raise ValueError(f"Unsupported file_list_rounding {rounding} in legacy reader")
        kwargs_legacy["file_list_include_preceding_frame"] = rounding == "start_down_end_up"
    if "pad_mode" in kwargs_legacy:
        pad_mode = kwargs_legacy.pop("pad_mode")
        if pad_mode not in ("none", "constant"):
            raise ValueError(f"Unsupported pad_mode {pad_mode} in legacy reader")
        kwargs_legacy["pad_sequences"] = pad_mode == "constant"
    return kwargs_legacy


def compare_experimental_to_legacy_reader(
    device, batch_size, experimental_only_kwargs=None, **kwargs
):
    """Runs legacy readers.video and experimental.readers.video side by side for one epoch and
    asserts that they produce the same samples.

    `kwargs` go to both readers (translated to legacy argument names); `experimental_only_kwargs`
    go to the experimental reader only.

    Checks equal epoch sizes, then every in-epoch sample of every output: video frames with
    compare_frames' tolerance (NVDEC and libavcodec differ slightly), labels and frame_num
    exactly, timestamps within 1 ms (the readers compute them in different time bases).
    """
    experimental_only_kwargs = experimental_only_kwargs or {}

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=1)
    def video_reader_pipeline():
        outs0 = fn.readers.video(
            device="gpu", name="legacy_reader", **_legacy_reader_kwargs(kwargs)
        )
        if isinstance(outs0, DataNode):
            outs0 = (outs0,)
        outs1 = fn.experimental.readers.video(
            device=device, name="experimental_reader", **kwargs, **experimental_only_kwargs
        )
        if isinstance(outs1, DataNode):
            outs1 = (outs1,)
        return tuple(list(outs0) + list(outs1))

    output_kinds = ["video"]
    if kwargs.get("labels") is not None or kwargs.get("file_list") or kwargs.get("file_root"):
        output_kinds.append("labels")
    if kwargs.get("enable_frame_num", False) not in (False, "none"):
        output_kinds.append("frame_num")
    if kwargs.get("enable_timestamps", False):
        output_kinds.append("timestamps")

    # compare_frames' threshold is expressed in 8-bit intensity steps; normalized float output
    # lies in [0.0, 1.0], so scale it back to [0, 255] before comparing, otherwise no content
    # difference could ever exceed the threshold.
    normalized_float = kwargs.get("dtype") == types.FLOAT and kwargs.get("normalized", False)
    value_scale = 255.0 if normalized_float else 1.0

    pipe = video_reader_pipeline()
    pipe.build()
    legacy_epoch_size = pipe.reader_meta("legacy_reader")["epoch_size"]
    experimental_epoch_size = pipe.reader_meta("experimental_reader")["epoch_size"]
    assert legacy_epoch_size == experimental_epoch_size, (
        f"Different epoch size: legacy {legacy_epoch_size}, "
        f"experimental {experimental_epoch_size}"
    )
    epoch_size = legacy_epoch_size

    num_iterations = (epoch_size + batch_size - 1) // batch_size
    for i in range(num_iterations):
        outs = pipe.run()
        assert len(outs) == 2 * len(
            output_kinds
        ), f"Expected outputs {output_kinds} from each reader, got {len(outs)} outputs in total"
        outs_legacy = outs[: len(output_kinds)]
        outs_experimental = outs[len(output_kinds) :]
        # The last batch of the epoch is filled with samples of the next epoch; skip those.
        num_valid = min(batch_size, epoch_size - i * batch_size)
        for kind, out_legacy, out_experimental in zip(output_kinds, outs_legacy, outs_experimental):
            for j in range(num_valid):
                sample_legacy = np.array(out_legacy[j].as_cpu())
                sample_experimental = np.array(out_experimental[j].as_cpu())
                where = f"iteration {i}, sample {j}, output '{kind}'"
                assert sample_legacy.shape == sample_experimental.shape, (
                    f"Shape mismatch at {where}: legacy {sample_legacy.shape}, "
                    f"experimental {sample_experimental.shape}"
                )
                if kind == "video":
                    for k in range(sample_legacy.shape[0]):
                        compare_frames(
                            np.float32(sample_experimental[k]) * value_scale,
                            np.float32(sample_legacy[k]) * value_scale,
                            i,
                            j,
                            k,
                        )
                elif kind == "timestamps":
                    np.testing.assert_allclose(
                        np.float64(sample_experimental),
                        np.float64(sample_legacy),
                        atol=1e-3,
                        err_msg=where,
                    )
                else:
                    np.testing.assert_array_equal(
                        sample_experimental.flatten(), sample_legacy.flatten(), err_msg=where
                    )


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


def test_file_list_include_end_actually_includes_end_frame():
    # Deterministic (non-randomized) boundary test: a file_list entry with
    # file_list_format="frames", start=0, end=2 should yield 3 frames (0, 1, 2)
    # when file_list_include_end=True, and 2 frames (0, 1) when False.
    #
    # sequence_length is chosen to exactly match the expected frame count for each
    # branch: with pad_mode="none" (ISOLATED boundary), a range shorter than
    # sequence_length produces zero samples rather than a short/padded one, so
    # sequence_length must equal the exact number of frames being pinned down.
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    list_file.write(f"{VIDEO_FILES[0]} 0 0 2\n")
    list_file.close()

    def run(include_end, sequence_length):
        @pipeline_def(batch_size=1, num_threads=3, device_id=0)
        def pipe():
            video, _label = fn.experimental.readers.video(
                device="cpu",
                file_list=list_file.name,
                file_list_format="frames",
                file_list_rounding="all_down",
                file_list_include_end=include_end,
                sequence_length=sequence_length,
                pad_mode="none",
            )
            return video

        p = pipe()
        p.build()
        (video,) = p.run()
        return np.array(video[0]).shape[0]

    assert run(include_end=True, sequence_length=3) == 3
    assert run(include_end=False, sequence_length=2) == 2


def test_file_list_omitted_end_means_end_of_video():
    list_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    list_file.write(f"{VIDEO_FILES[0]} 0 2\n")  # label=0, start=2, end omitted (parses as 0)
    list_file.close()

    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        video, _label = fn.experimental.readers.video(
            device="cpu",
            file_list=list_file.name,
            file_list_format="frames",
            sequence_length=1,
            pad_mode="none",
        )
        return video

    p = pipe()
    p.build()
    (video,) = p.run()
    # Must decode from frame 2 through the end of the video, not zero frames.
    assert np.array(video[0]).shape[0] > 0


def _file_list_frame_selection(reader_fn, file_list, **kwargs):
    """Returns (epoch_size, sorted start frame indices) for a single-frame-per-sample reader."""

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        _, _, frame_num = reader_fn(
            name="reader",
            file_list=file_list,
            sequence_length=1,
            step=1,
            enable_frame_num=True,
            **kwargs,
        )
        return frame_num

    p = pipe()
    p.build()
    epoch_size = p.reader_meta("reader")["epoch_size"]
    frame_nums = []
    for _ in range(epoch_size):
        (frame_num,) = p.run()
        frame_nums.append(int(np.array(frame_num.as_cpu()[0]).flatten()[0]))
    return epoch_size, sorted(frame_nums)


_file_list_dir = tempfile.mkdtemp(prefix="dali_video_reader_file_lists_")
atexit.register(shutil.rmtree, _file_list_dir, ignore_errors=True)


def _write_file_list(lines):
    """Writes the given file_list entries (one string per line) to a new temporary file and
    returns its path."""
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


@cartesian_params(devices, file_list_formats)
def test_file_list_default_end_matches_legacy(device, file_list_format):
    # Deterministic check that, without any `file_list_include_end` override,
    # experimental.readers.video selects exactly the same frames from a file_list with explicit
    # start/end values as the legacy readers.video does (i.e. `end` is exclusive by default).
    video_files = sorted(VIDEO_FILES)
    if file_list_format == "frames":
        entries = [(video_files[0], 5, 15), (video_files[1], 0, 10), (video_files[2], 3, 7)]
    else:
        entries = [(video_files[0], 0.2, 0.8), (video_files[1], 0.1, 0.9)]

    for filename, start, end in entries:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as list_file:
            list_file.write(f"{filename} 0 {start} {end}\n")
            list_file.flush()

            # DALI-4917: experimental's file_list_rounding default ("start_up_end_down") now
            # matches legacy's file_list_include_preceding_frame default (False), so no
            # explicit rounding override is needed for either format any more.
            legacy_kwargs = dict(file_list_frame_num=(file_list_format == "frames"))
            experimental_kwargs = dict(file_list_format=file_list_format)

            legacy = _file_list_frame_selection(
                functools.partial(fn.readers.video, device="gpu"),
                list_file.name,
                **legacy_kwargs,
            )
            experimental = _file_list_frame_selection(
                functools.partial(fn.experimental.readers.video, device=device),
                list_file.name,
                **experimental_kwargs,
            )
            assert legacy == experimental, (
                f"Frame selection mismatch for {filename} [{start}, {end}] "
                f"({file_list_format}): legacy (epoch_size, frames)={legacy}, "
                f"experimental={experimental}"
            )
            if file_list_format == "frames":
                # Exclusive end: frames start..end-1.
                assert experimental == (end - start, list(range(start, end)))


@cartesian_params(devices)
def test_file_list_end_beyond_video_length_is_clamped(device):
    # Regression test: with the default file_list_include_end=False, a file_list `end` value
    # larger than the video's actual frame count must be clamped to the video's frame count,
    # not used verbatim. Previously (before this fix) the num_frames clamp only ran inside the
    # `if (include_end)` branch of PrepareMetadataImpl in video_reader_decoder_op.h, so default
    # users got no clamp at all: a `5 100000` entry on a 240-frame video produced a bogus
    # epoch_size of 99995 and crashed partway through the epoch ("Unexpected out-of-bounds frame
    # index ... for pad_mode = 'none'"). Legacy readers.video rejects such input at build time
    # instead; experimental.readers.video should at least clamp to a valid, decodable range.
    video_file = sorted(VIDEO_FILES)[0]

    # Determine the video's actual frame count.
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def count_pipe():
        video = fn.experimental.readers.video(
            device=device, filenames=[video_file], sequence_length=1, pad_mode="none", name="r"
        )
        return video

    p = count_pipe()
    p.build()
    num_frames = p.reader_meta("r")["epoch_size"]

    start = 5
    huge_end = 100000
    assert huge_end > num_frames, "Test assumption: huge_end must exceed the video's frame count"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as list_file:
        list_file.write(f"{video_file} 0 {start} {huge_end}\n")
        list_file.flush()

        epoch_size, frame_nums = _file_list_frame_selection(
            functools.partial(fn.experimental.readers.video, device=device),
            list_file.name,
            file_list_format="frames",
        )

        # end_frame should clamp to num_frames (no +1, since include_end defaults to False), so
        # the epoch covers exactly [start, num_frames).
        assert epoch_size == num_frames - start, (
            f"Expected epoch_size={num_frames - start} (clamped to video length), "
            f"got {epoch_size}"
        )
        assert frame_nums == list(range(start, num_frames))


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


_dtype_cases = {
    "uint8": dict(dtype=types.UINT8, normalized=False),
    "float_normalized": dict(dtype=types.FLOAT, normalized=True),
    "float_unnormalized": dict(dtype=types.FLOAT, normalized=False),
}


@cartesian_params(devices, batch_sizes, sequence_lengths, list(_dtype_cases.keys()))
def test_compare_experimental_to_legacy_reader_dtype(
    device, batch_size, sequence_length, dtype_case
):
    dtype_kwargs = _dtype_cases[dtype_case]
    if device == "cpu" and dtype_kwargs["dtype"] == types.FLOAT:
        # dtype=FLOAT is only supported on the GPU backend of experimental.readers.video
        # (see test_dtype_float_on_cpu_raises); not a legacy-vs-experimental parity gap.
        raise SkipTest("dtype=FLOAT is not supported on the CPU backend")
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        filenames=VIDEO_FILES,
        sequence_length=sequence_length,
        **dtype_kwargs,
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


@cartesian_params([True, False])
def test_dtype_normalized(normalized):
    # dtype=FLOAT is only supported on the GPU backend (see test_dtype_float_on_cpu_raises);
    # unlike most experimental.readers.video tests, this one is GPU-only.
    device = "gpu"

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device=device,
            filenames=VIDEO_FILES,
            sequence_length=3,
            dtype=types.FLOAT,
            normalized=normalized,
        )

    p = pipe()
    p.build()
    (video,) = p.run()
    sample = np.array(video[0].as_cpu())
    assert sample.dtype == np.float32
    if normalized:
        # Chroma<->RGB conversion introduces small floating-point overshoot/undershoot
        # around the [0, 1] boundary (matching the tolerance used by the C++
        # FramesDecoderGpuTest.NormalizedFloatOutputInRange/UnnormalizedFloatOutputInByteRange
        # tests for the same conversion path), so allow a small epsilon rather than a hard
        # [0, 1] bound.
        epsilon = 0.25
        in_range = np.logical_and(sample >= -epsilon, sample <= 1.0 + epsilon)
        assert in_range.mean() > 0.97, "Expected almost all pixels in [0, 1] (+/- epsilon)"
        assert np.any(np.logical_and(sample >= 0.1, sample <= 0.9)), (
            "Expected to find pixel values within the normalized range, "
            "confirming normalization actually happened"
        )
    else:
        assert sample.max() > 1.0, "Expected unnormalized float output above 1.0"


@cartesian_params(devices)
def test_channels_arg_matches_decoder_channels(device):
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device=device, filenames=VIDEO_FILES, sequence_length=3, channels=3
        )

    p = pipe()
    p.build()
    p.run()  # should not raise


def test_channels_arg_mismatch_raises():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="cpu", filenames=VIDEO_FILES, sequence_length=3, channels=4
        )

    try:
        pipe().build()
        assert False, "Expected an exception for channels=4 (decoder always produces 3 channels)"
    except RuntimeError:
        pass  # expected


def test_dtype_float_on_cpu_raises():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="cpu", filenames=VIDEO_FILES, sequence_length=3, dtype=types.FLOAT
        )

    try:
        pipe().build()
        assert False, "Expected an exception for dtype=FLOAT on the CPU backend"
    except RuntimeError:
        pass  # expected


# VFR_VIDEO_FILE/CFR_VIDEO_FILE are h264, which the CPU variant of this operator does not
# support (see frames_decoder_cpu.cc), so these tests are GPU-only, matching the way VFR
# content is exercised in the legacy C++ gtest suite (VariableFrameRate/VariableFrameRate2 in
# video_reader_op_test.cc).
def test_require_constant_frame_rate_default_is_permissive():
    # Default behavior (require_constant_frame_rate=False) must remain unchanged: VFR content
    # decodes without error, exactly as before this argument existed.
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[VFR_VIDEO_FILE], sequence_length=3
        )

    p = pipe()
    p.build()
    p.run()  # must not raise


def test_require_constant_frame_rate_accepts_cfr():
    # A constant frame rate video must not be rejected when the strict check is enabled.
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu",
            filenames=[CFR_VIDEO_FILE],
            sequence_length=3,
            require_constant_frame_rate=True,
        )

    p = pipe()
    p.build()
    p.run()  # must not raise


def test_require_constant_frame_rate_rejects_vfr():
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu",
            filenames=[VFR_VIDEO_FILE],
            sequence_length=3,
            require_constant_frame_rate=True,
        )

    try:
        p = pipe()
        p.build()
        p.run()
        assert False, "Expected an exception for a variable frame rate video"
    except RuntimeError:
        pass  # expected


def test_require_constant_frame_rate_rejects_vfr_on_cache_hit():
    # FrameIndexCache caches the per-filename decode index across operator instances. Warm the
    # cache first with a permissive reader (require_constant_frame_rate=False), then build a
    # second, independent operator instance for the same file with the strict check enabled:
    # it must still raise, exercising the SetIndex()/cache-hit path rather than the fresh
    # BuildIndex() path.
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def warm_pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[VFR_VIDEO_FILE], sequence_length=3
        )

    warm = warm_pipe()
    warm.build()
    warm.run()  # must not raise; also populates FrameIndexCache for VFR_VIDEO_FILE

    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def strict_pipe():
        return fn.experimental.readers.video(
            device="gpu",
            filenames=[VFR_VIDEO_FILE],
            sequence_length=3,
            require_constant_frame_rate=True,
        )

    try:
        strict = strict_pipe()
        strict.build()
        strict.run()
        assert False, "Expected an exception for a variable frame rate video (cache-hit path)"
    except RuntimeError:
        pass  # expected


def test_additional_decode_surfaces_does_not_crash():
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=VIDEO_FILES, sequence_length=3, additional_decode_surfaces=4
        )

    p = pipe()
    p.build()
    p.run()


def test_additional_decode_surfaces_matches_default_output():
    # additional_decode_surfaces only sizes the GPU decoder's decode-lookahead / reorder buffer;
    # it must not change the decoded pixel data, so decoding the same video with a non-default
    # value should produce identical output to the default (additional_decode_surfaces=2).
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipe(additional_decode_surfaces):
        return fn.experimental.readers.video(
            device="gpu",
            filenames=VIDEO_FILES,
            sequence_length=3,
            additional_decode_surfaces=additional_decode_surfaces,
        )

    p_default = pipe(2)
    p_default.build()
    (video_default,) = p_default.run()

    p_custom = pipe(6)
    p_custom.build()
    (video_custom,) = p_custom.run()

    for sample_default, sample_custom in zip(video_default, video_custom):
        np.testing.assert_array_equal(
            np.array(sample_default.as_cpu()), np.array(sample_custom.as_cpu())
        )


def test_additional_decode_surfaces_negative_raises():
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=VIDEO_FILES, sequence_length=3, additional_decode_surfaces=-1
        )

    try:
        pipe().build()
        assert False, "Expected an exception for a negative additional_decode_surfaces"
    except RuntimeError:
        pass  # expected


@cartesian_params([None, types.UINT8, types.FLOAT])
def test_output_dtype_metadata(dtype):
    # The schema must statically report the video output's dtype (derived from the `dtype`
    # argument), so that graph-level metadata consumers see it before the operator runs.
    from nvidia.dali import backend

    spec = backend.OpSpec("experimental__readers__Video")
    spec.AddArg("device", "gpu")
    spec.AddArg("sequence_length", 3)
    if dtype is not None:
        spec.AddArg("dtype", dtype)
    spec.AddOutput("video", "gpu")
    spec.InferOutputMetadata()
    output_dtype = spec.OutputDesc(0)[3]
    expected = types.UINT8 if dtype is None else dtype
    assert output_dtype == expected, f"Expected {expected}, got {output_dtype}"


_LEGACY_ONLY_ARGS = (
    "pad_sequences",
    "file_list_frame_num",
    "file_list_include_preceding_frame",
    "skip_vfr_check",
)


@cartesian_params(devices)
def test_legacy_pad_sequences_maps_to_pad_mode_constant(device):
    # 7-frame range, sequence_length=5 (default step=5): frames 0-4 form a full sequence and
    # frames 5-6 only fit a padded one.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 7"])
    experimental = functools.partial(
        fn.experimental.readers.video,
        device=device,
        file_list=list_file,
        file_list_format="frames",
        sequence_length=5,
    )
    with assert_warns(DeprecationWarning, glob="*pad_sequences*"):
        via_legacy_arg = _collect_samples(experimental, pad_sequences=True)
    via_new_arg = _collect_samples(experimental, pad_mode="constant")
    assert (
        via_legacy_arg == via_new_arg == (2, [(0, 0, 5), (0, 5, 5)])
    ), f"pad_sequences=True: {via_legacy_arg}, pad_mode='constant': {via_new_arg}"
    assert _collect_samples(experimental, pad_sequences=False) == (1, [(0, 0, 5)])

    legacy = _collect_samples(
        functools.partial(
            fn.readers.video,
            device="gpu",
            file_list=list_file,
            file_list_frame_num=True,
            sequence_length=5,
        ),
        pad_sequences=True,
    )
    assert legacy == via_legacy_arg, f"legacy: {legacy}, experimental: {via_legacy_arg}"


@cartesian_params(devices)
def test_legacy_file_list_frame_num_maps_to_file_list_format(device):
    reader = functools.partial(fn.experimental.readers.video, device=device)

    frames_list = _write_file_list([f"{VIDEO_0} 0 5 15"])
    with assert_warns(DeprecationWarning, glob="*file_list_frame_num*"):
        via_legacy_arg = _file_list_frame_selection(reader, frames_list, file_list_frame_num=True)
    via_new_arg = _file_list_frame_selection(reader, frames_list, file_list_format="frames")
    assert via_legacy_arg == via_new_arg == (10, list(range(5, 15)))

    # 0.5 s and 1.5 s are exactly frames 12 and 36 at 24 fps; `end` is exclusive.
    seconds_list = _write_file_list([f"{VIDEO_0} 0 0.5 1.5"])
    via_legacy_arg = _file_list_frame_selection(reader, seconds_list, file_list_frame_num=False)
    via_new_arg = _file_list_frame_selection(reader, seconds_list, file_list_format="timestamps")
    assert via_legacy_arg == via_new_arg == (24, list(range(12, 36)))


@cartesian_params(devices)
def test_legacy_file_list_include_preceding_frame_maps_to_rounding(device):
    reader = functools.partial(fn.experimental.readers.video, device=device)

    # 0.52 s and 1.48 s fall between frames (12.48 and 35.52 at 24 fps).
    seconds_list = _write_file_list([f"{VIDEO_0} 0 0.52 1.48"])
    with assert_warns(DeprecationWarning, glob="*file_list_include_preceding_frame*"):
        preceding = _file_list_frame_selection(
            reader, seconds_list, file_list_include_preceding_frame=True
        )
    rounding = _file_list_frame_selection(
        reader, seconds_list, file_list_rounding="start_down_end_up"
    )
    assert preceding == rounding == (24, list(range(12, 36)))
    not_preceding = _file_list_frame_selection(
        reader, seconds_list, file_list_include_preceding_frame=False
    )
    assert not_preceding == (22, list(range(13, 35)))

    # Legacy documents that the flag has no effect when frame numbers are used: the default
    # rounding (start up, end down) must still apply to fractional frame numbers.
    frames_list = _write_file_list([f"{VIDEO_0} 0 5.5 14.5"])
    ignored = _file_list_frame_selection(
        reader, frames_list, file_list_frame_num=True, file_list_include_preceding_frame=True
    )
    default = _file_list_frame_selection(reader, frames_list, file_list_format="frames")
    assert ignored == default == (8, list(range(6, 14)))


@params(
    dict(pad_sequences=True, pad_mode="constant"),
    dict(pad_sequences=False, pad_mode="edge"),
    dict(file_list_frame_num=True, file_list_format="frames"),
    dict(file_list_include_preceding_frame=True, file_list_rounding="start_down_end_up"),
)
def test_legacy_arg_conflicts_with_new_arg(conflicting_kwargs):
    list_file = _write_file_list([f"{VIDEO_0} 0 0 30"])
    legacy_arg = next(k for k in conflicting_kwargs if k in _LEGACY_ONLY_ARGS)

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        video, label = fn.experimental.readers.video(
            device="cpu", file_list=list_file, sequence_length=3, **conflicting_kwargs
        )
        return video, label

    with assert_raises(RuntimeError, glob=f"*{legacy_arg}*cannot be combined*"):
        pipe().build()


def test_legacy_arg_conflicts_with_new_arg_without_file_list():
    # The legacy-vs-new arg conflict check must fire regardless of which file source argument
    # is used, not just when `file_list` is given (it is implemented inside GetFileListOptions,
    # which used to only be called when `file_list` was non-empty).
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="cpu",
            filenames=[VIDEO_0],
            sequence_length=3,
            file_list_frame_num=True,
            file_list_format="frames",
        )

    with assert_raises(RuntimeError, glob="*file_list_frame_num*cannot be combined*"):
        pipe().build()


@params(True, False)
def test_legacy_skip_vfr_check_is_accepted_and_ignored(skip_vfr_check):
    # skip_vfr_check is a pure no-op: in particular skip_vfr_check=False must NOT turn on
    # require_constant_frame_rate, so the VFR file still decodes.
    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="gpu",
            filenames=[VFR_VIDEO_FILE],
            sequence_length=3,
            skip_vfr_check=skip_vfr_check,
        )

    with assert_warns(DeprecationWarning, glob="*skip_vfr_check*"):
        p = pipe()
        p.build()
    p.run()  # must not raise


@cartesian_params(devices)
def test_stride_sequence_fits_when_last_frame_exists(device):
    # sequence_length=3, stride=4 uses frames s, s+4, s+8: a 9-frame span. Legacy only requires
    # the last frame to exist, so a 9-frame range holds exactly one sequence. The second, longer
    # entry keeps the dataset non-empty on the unfixed reader.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 9", f"{VIDEO_0} 1 0 30"])
    kwargs = dict(file_list=list_file, sequence_length=3, stride=4)
    experimental = _collect_samples(
        functools.partial(fn.experimental.readers.video, device=device, file_list_format="frames"),
        **kwargs,
    )
    legacy = _collect_samples(
        functools.partial(fn.readers.video, device="gpu", file_list_frame_num=True), **kwargs
    )
    expected = (3, [(0, 0, 3), (1, 0, 3), (1, 12, 3)])
    assert legacy == expected, f"legacy: {legacy}"
    assert experimental == expected, f"experimental: {experimental}, legacy: {legacy}"


_legacy_gpu_reader = functools.partial(fn.readers.video, device="gpu")


def _experimental_reader(device):
    return functools.partial(fn.experimental.readers.video, device=device)


def _import_av_or_skip():
    try:
        import av
    except ImportError:
        raise SkipTest("PyAV (`av`) is required to build the shifted-pts fixture")
    return av


def _make_pts_shifted_copy(src, offset_seconds, tmp_dir):
    """Remuxes (no re-encoding) the video stream of `src` into an MP4 file in `tmp_dir` whose
    timestamps are all shifted by `offset_seconds`; returns the new file's path.

    MP4 is used (rather than e.g. Matroska) because its header carries the track's frame rate
    and start time explicitly (mdhd/edit-list boxes), so both the legacy and experimental readers
    -- which only parse container headers, not deep per-packet probing -- pick up the shifted
    start time correctly. A plain `add_stream_from_template` remux to Matroska was tried first:
    ffmpeg's Matroska muxer/demuxer round-trip neither the per-track default frame duration nor a
    nonzero stream start time through header-only parsing, which made the legacy reader compute
    a wrong (zero) start time and either miscount frames or fail to seek.
    """
    av = _import_av_or_skip()
    dst = os.path.join(tmp_dir, "shifted.mp4")
    with av.open(src) as inp, av.open(dst, mode="w", format="mp4") as out:
        in_stream = inp.streams.video[0]
        out_stream = out.add_stream(
            codec_name=in_stream.codec_context.name, rate=in_stream.average_rate
        )
        out_stream.codec_context.extradata = in_stream.codec_context.extradata
        out_stream.codec_context.width = in_stream.codec_context.width
        out_stream.codec_context.height = in_stream.codec_context.height
        out_stream.codec_context.pix_fmt = in_stream.codec_context.pix_fmt
        out_stream.codec_context.time_base = in_stream.time_base
        offset = int(round(offset_seconds / in_stream.time_base))
        for packet in inp.demux(in_stream):
            if packet.dts is None:  # demuxer flush packet
                continue
            packet.pts += offset
            packet.dts += offset
            packet.stream = out_stream
            out.mux(packet)
    with av.open(dst) as check:
        stream = check.streams.video[0]
        first_pts = min(p.pts for p in check.demux(stream) if p.pts is not None)
        first_seconds = float(first_pts * stream.time_base)
        assert first_seconds >= offset_seconds - 0.01, (
            f"Fixture precondition: expected the first pts at >= {offset_seconds} s, "
            f"got {first_seconds} s"
        )
    return dst


def test_video_0_fixture_assumptions():
    # The file_list tests hard-code frame numbers for VIDEO_0: 240 frames at 24 fps.
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            device="cpu", filenames=[VIDEO_0], sequence_length=1, name="r"
        )

    p = pipe()
    p.build()
    assert p.reader_meta("r")["epoch_size"] == 240


@cartesian_params(devices)
def test_file_list_timestamps_omitted_end_means_end_of_video(device):
    list_file = _write_file_list([f"{VIDEO_0} 0 2.5"])  # end omitted: 2.5 s (frame 60) to the end
    experimental = _file_list_frame_selection(_experimental_reader(device), list_file)
    legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file)
    assert (
        experimental == legacy == (180, list(range(60, 240)))
    ), f"experimental: {experimental[0]} frames, legacy: {legacy[0]} frames"


@cartesian_params(devices)
def test_file_list_timestamps_end_at_duration_is_kept(device):
    list_file = _write_file_list([f"{VIDEO_0} 0 1.0 10.0"])  # end == video duration
    experimental = _file_list_frame_selection(_experimental_reader(device), list_file)
    legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file)
    assert (
        experimental == legacy == (216, list(range(24, 240)))
    ), f"experimental: {experimental[0]} frames, legacy: {legacy[0]} frames"


@cartesian_params(devices)
def test_file_list_timestamps_start_past_end_of_video_raises(device):
    # The valid first entry keeps the dataset non-empty, so the only way to fail is to reject
    # the second one. `end` is given (and past the duration too) so that legacy's own
    # start<=end check does not trip before its start<=duration check: with end omitted, legacy
    # would instead report "Start time number should be lesser or equal to end time" because it
    # substitutes end=frame_count for an omitted end before comparing.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 1.0", f"{VIDEO_0} 1 10.5 20"])
    with assert_raises(RuntimeError, glob="*past the end of the video*"):
        _file_list_frame_selection(_experimental_reader(device), list_file)
    with assert_raises(RuntimeError, glob="*greater than video duration*"):
        _file_list_frame_selection(_legacy_gpu_reader, list_file)


@cartesian_params(devices)
def test_file_list_timestamps_start_at_end_of_video_is_skipped_not_error(device):
    # Legacy accepts start == duration (ceil(10.0 * 24) == frame count) and yields no samples
    # for that entry.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 1.0", f"{VIDEO_0} 1 10.0"])
    experimental = _file_list_frame_selection(_experimental_reader(device), list_file)
    legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file)
    assert experimental == legacy == (24, list(range(0, 24)))


@cartesian_params(devices)
def test_file_list_timestamps_negative_count_from_end(device):
    # 2 s and 1 s before the end of a 240-frame 24 fps video: frames 192..215.
    list_file = _write_file_list([f"{VIDEO_0} 0 -2.0 -1.0"])
    experimental = _file_list_frame_selection(_experimental_reader(device), list_file)
    legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file)
    assert (
        experimental == legacy == (24, list(range(192, 216)))
    ), f"experimental: {experimental}, legacy: {legacy}"


@cartesian_params(devices)
def test_file_list_timestamps_are_relative_to_stream_start(device):
    with tempfile.TemporaryDirectory() as tmp_dir:
        shifted = _make_pts_shifted_copy(CFR_VP9_60FPS_FILE, 1.0, tmp_dir)

        # 0.21 s and 0.49 s after the first frame fall between frames (12.6 and 29.4 at 60 fps),
        # so the selection is frames 13..28 with the default rounding.
        list_file = _write_file_list([f"{shifted} 0 0.21 0.49"])
        experimental = _file_list_frame_selection(_experimental_reader(device), list_file)
        legacy = _file_list_frame_selection(_legacy_gpu_reader, list_file)
        assert (
            experimental == legacy == (16, list(range(13, 29)))
        ), f"experimental: {experimental}, legacy: {legacy}"

        # A negative start counts back from the end of the stream regardless of the first pts:
        # 0.29 s before the end of a 50-frame 60 fps video is frame 32.6, rounded up to 33.
        negative_list = _write_file_list([f"{shifted} 0 -0.29"])
        negative = _file_list_frame_selection(_experimental_reader(device), negative_list)
        assert negative == (17, list(range(33, 50))), f"experimental: {negative}"


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


_float_fill_cases = {
    "default_fill": (dict(normalized=False), [0.0, 0.0, 0.0]),
    "normalized_255": (dict(normalized=True, fill_value=[255]), [1.0, 1.0, 1.0]),
    "per_channel": (dict(normalized=False, fill_value=[10, 20, 30]), [10.0, 20.0, 30.0]),
}


@params(*_float_fill_cases.keys())
def test_float_constant_padding(case):
    # dtype=FLOAT is GPU-only. 7-frame range with sequence_length=5: the second sample holds
    # frames 5 and 6 followed by 3 padded frames.
    extra_kwargs, expected_pixel = _float_fill_cases[case]
    list_file = _write_file_list([f"{VIDEO_0} 0 0 7"])

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def pipe():
        video, _, frame_num = fn.experimental.readers.video(
            device="gpu",
            file_list=list_file,
            file_list_format="frames",
            sequence_length=5,
            dtype=types.FLOAT,
            pad_mode="constant",
            enable_frame_num="sequence",
            **extra_kwargs,
        )
        return video, frame_num

    p = pipe()
    p.build()
    p.run()  # first sample: frames 0-4, no padding
    video, frame_num = p.run()
    frames = np.array(video.as_cpu()[0])
    assert frames.dtype == np.float32
    assert list(np.array(frame_num.as_cpu()[0])) == [5, 6, -1, -1, -1]
    padded = frames[2:]
    np.testing.assert_array_equal(padded, np.broadcast_to(np.float32(expected_pixel), padded.shape))


def test_float_constant_padding_matches_legacy_zero_padding():
    list_file = _write_file_list([f"{VIDEO_0} 0 0 7"])

    def second_sample(reader_fn, **kwargs):
        @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
        def pipe():
            video, _ = reader_fn(
                device="gpu", file_list=list_file, sequence_length=5, dtype=types.FLOAT, **kwargs
            )
            return video

        p = pipe()
        p.build()
        p.run()
        (video,) = p.run()
        return np.array(video.as_cpu()[0])

    legacy = second_sample(fn.readers.video, file_list_frame_num=True, pad_sequences=True)
    experimental = second_sample(
        fn.experimental.readers.video, file_list_format="frames", pad_mode="constant"
    )
    assert legacy.shape == experimental.shape
    np.testing.assert_array_equal(experimental[2:], legacy[2:])


def test_pad_sequences_matches_pad_mode_constant_with_float_dtype():
    # pad_sequences=True is legacy-compat sugar for pad_mode="constant" (see GetReaderBoundaryType
    # in video_reader_decoder_op.cc); check that equivalence still holds when combined with
    # dtype=FLOAT, which neither of the two dedicated tests above exercises together.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 7"])

    def second_sample(**extra_kwargs):
        @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
        def pipe():
            video, _ = fn.experimental.readers.video(
                device="gpu",
                file_list=list_file,
                file_list_format="frames",
                sequence_length=5,
                dtype=types.FLOAT,
                **extra_kwargs,
            )
            return video

        p = pipe()
        p.build()
        p.run()  # first sample: frames 0-4, no padding
        (video,) = p.run()
        return np.array(video.as_cpu()[0])

    via_pad_sequences = second_sample(pad_sequences=True)
    via_pad_mode = second_sample(pad_mode="constant")
    np.testing.assert_array_equal(via_pad_sequences, via_pad_mode)


@cartesian_params(devices, [1, 2], ["constant", "edge"])
def test_padding_emits_sample_at_every_remaining_step(device, stride, pad_mode):
    # 10-frame range, sequence_length=3, step=1: full sequences start at 0..7 (stride 1) or
    # 0..5 (stride 2); with padding, legacy also emits a padded sample at every remaining start,
    # i.e. one sample per frame of the range.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 10"])
    kwargs = dict(file_list=list_file, sequence_length=3, stride=stride, step=1)
    expected = (10, [(0, s, 3) for s in range(10)])
    experimental = _collect_samples(
        functools.partial(
            fn.experimental.readers.video,
            device=device,
            file_list_format="frames",
            pad_mode=pad_mode,
        ),
        **kwargs,
    )
    assert experimental == expected, f"experimental: {experimental}"
    if pad_mode == "constant":
        legacy = _collect_samples(
            functools.partial(
                fn.readers.video, device="gpu", file_list_frame_num=True, pad_sequences=True
            ),
            **kwargs,
        )
        assert legacy == expected, f"legacy: {legacy}"


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


# Builds a CPU-only experimental.readers.video pipeline in a child process whose address space
# is capped, so that an infinite sample-generation loop cannot exhaust the machine's memory.
_BOUNDED_BUILD_CHILD = r"""
import resource
import sys

limit = 6 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

from nvidia.dali import fn, pipeline_def


@pipeline_def(batch_size=1, num_threads=1, device_id=None)
def pipe():
    return fn.experimental.readers.video(
        device="cpu",
        filenames=[sys.argv[1]],
        sequence_length=int(sys.argv[2]),
        stride=int(sys.argv[3]),
    )


try:
    pipe().build()
except Exception as e:
    print("RAISED:", e)
    sys.exit(0)
print("BUILT")
"""


def _build_reader_in_bounded_subprocess(sequence_length, stride, timeout=120):
    try:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _BOUNDED_BUILD_CHILD,
                VIDEO_0,
                str(sequence_length),
                str(stride),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"build() with sequence_length={sequence_length}, stride={stride} did not return "
            f"within {timeout} s (sample-generation loop not advancing?)"
        )


def test_stride_zero_raises_instead_of_hanging():
    res = _build_reader_in_bounded_subprocess(sequence_length=3, stride=0)
    assert "RAISED:" in res.stdout and "stride" in res.stdout, (
        f"Expected a prompt error mentioning `stride`; returncode={res.returncode}, "
        f"stdout={res.stdout!r}, stderr tail={res.stderr[-2000:]!r}"
    )


def test_sequence_length_zero_raises_instead_of_hanging():
    res = _build_reader_in_bounded_subprocess(sequence_length=0, stride=1)
    assert "RAISED:" in res.stdout and "sequence_length" in res.stdout, (
        f"Expected a prompt error mentioning `sequence_length`; returncode={res.returncode}, "
        f"stdout={res.stdout!r}, stderr tail={res.stderr[-2000:]!r}"
    )


def test_harness_detects_epoch_size_mismatch():
    # file_list_include_end=True makes experimental select 11 frames where legacy selects 10.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 10"])
    with assert_raises(AssertionError, glob="*epoch size*"):
        compare_experimental_to_legacy_reader(
            device="gpu",
            batch_size=1,
            file_list=list_file,
            file_list_format="frames",
            sequence_length=1,
            experimental_only_kwargs=dict(file_list_include_end=True),
        )


def test_harness_compares_samples_after_the_first():
    # Legacy pads with zeros; fill_value=255 makes experimental pad with 255. Only the second
    # sample (iteration 1) contains padding, so this is caught only if the harness compares
    # past the first sample.
    list_file = _write_file_list([f"{VIDEO_0} 0 0 7"])
    with assert_raises(AssertionError, glob="*differs from reference*"):
        compare_experimental_to_legacy_reader(
            device="gpu",
            batch_size=1,
            file_list=list_file,
            file_list_format="frames",
            sequence_length=5,
            pad_mode="constant",
            experimental_only_kwargs=dict(fill_value=[255]),
        )


@cartesian_params(devices, [1, 3], [-1, 1, 4], pad_modes_supported_by_legacy_reader)
def test_compare_experimental_to_legacy_reader_stride_step(device, stride, step, pad_mode):
    list_file = _write_file_list([f"{VIDEO_0} 0 0 20", f"{VIDEO_0} 1 5 17"])
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=4,
        file_list=list_file,
        file_list_format="frames",
        sequence_length=3,
        stride=stride,
        step=step,
        pad_mode=pad_mode,
        enable_frame_num=True,
        enable_timestamps=True,
    )


@cartesian_params(devices, batch_sizes)
def test_compare_experimental_to_legacy_reader_file_list_timestamp_edges(device, batch_size):
    list_file = _write_file_list(
        [
            f"{VIDEO_0} 0 9.0",  # omitted end
            f"{VIDEO_0} 1 8.5 10.0",  # end == duration
            f"{VIDEO_0} 2 -1.0 -0.5",  # negative, counted from the end
        ]
    )
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=batch_size,
        file_list=list_file,
        file_list_format="timestamps",
        sequence_length=3,
        enable_frame_num=True,
        enable_timestamps=True,
    )


@cartesian_params(devices)
def test_compare_experimental_to_legacy_reader_empty_labels(device):
    compare_experimental_to_legacy_reader(
        device=device,
        batch_size=2,
        filenames=sorted(VIDEO_FILES)[:2],
        labels=[],
        sequence_length=3,
        step=100,
        enable_frame_num=True,
    )


# Annex-B (start-code delimited) H.264/HEVC fixtures from DALI_extra and their frame counts
# (see db/video/containers/README.rst and db/video/cfr/README.txt). h264_in_mpeg_ps is
# deliberately not included here: MPEG-PS is rejected outright (see
# test_mpeg_ps_is_rejected_as_unsupported) rather than reaching the Annex-B parsing this test
# targets, so it no longer belongs in this set.
ANNEX_B_VIDEOS = {
    "h264_in_avi": (DALI_EXTRA_PATH + "/db/video/containers/avi/cfr.avi", 180),
    "raw_h264": (DALI_EXTRA_PATH + "/db/video/cfr/test_1.h264", 50),
    "raw_h265": (DALI_EXTRA_PATH + "/db/video/cfr/test_1.h265", 50),
}


def test_mpeg_ps_is_rejected_as_unsupported():
    """MPEG-PS (container format 'mpeg') only stamps a timestamp on some packets, which the
    frame-accurate seek index experimental.readers.video builds can't handle reliably.
    FramesDecoderBase::BuildIndex() rejects it explicitly, before doing any index-building work.
    This only affects operators that need a seek index (this reader); sequential-decode-only
    operators such as experimental.decoders.video and experimental.inputs.video never call
    BuildIndex() and are unaffected. At the dataset-metadata-building stage, BuildIndex()'s
    rejection is caught and treated like any other invalid file: a file list mixing MPEG-PS with
    a working file excludes only the MPEG-PS entry, with a warning."""
    mpeg_ps_file = DALI_EXTRA_PATH + "/db/video/containers/mpeg/cfr.mpeg"
    avi_file = DALI_EXTRA_PATH + "/db/video/containers/avi/cfr.avi"

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def alone_pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[mpeg_ps_file], sequence_length=1
        )

    with assert_raises(RuntimeError, glob="*number of input samples: 0*"):
        p = alone_pipe()
        p.build()

    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def mixed_pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[mpeg_ps_file, avi_file], sequence_length=1, name="r"
        )

    p = mixed_pipe()
    p.build()
    assert p.reader_meta("r")["epoch_size"] == 180, "expected only the avi file's 180 frames"
    p.run()


@params(*ANNEX_B_VIDEOS.keys())
def test_annex_b_seek_matches_sequential_decode(case):
    """Detection test: experimental.readers.video builds its keyframe index by parsing H.264/HEVC
    NAL units as 4-byte length-prefixed (AVCC) in FramesDecoderBase::BuildIndex; Annex-B streams
    use start codes. Correct behavior: all frames are indexed, and a frame decoded after a seek
    is identical to the same frame decoded sequentially."""
    if case in ("raw_h264", "raw_h265"):
        raise SkipTest(
            "Known limitation (not the Annex-B NAL parsing bug this test targets, which is "
            "fixed): these are bare elementary streams with no container-level timestamps at "
            "all (every packet's pts/dts is AV_NOPTS_VALUE from the very first packet, so there "
            "is no baseline to interpolate from). This is a shared limitation, not a parity "
            "gap: legacy readers.video also fails to open these files, hitting its own "
            "variable-frame-rate heuristic check because it can't determine a reliable "
            "avg_frame_rate without container timing either"
        )
    path, expected_frames = ANNEX_B_VIDEOS[case]
    _check_seek_matches_sequential_decode(case, path, expected_frames)


# AVCC/ISO (length-prefixed) H.264/HEVC fixtures from DALI_extra and their frame counts. Each
# has NAL units whose 4-byte length prefix is 00 00 01 XX (a length in [256, 511]), which is
# byte-for-byte an Annex-B start code; test_25fps.mp4's only IDR slice and many of
# sintel_trailer-720p.mp4's IRAP pictures were missed when BuildIndex guessed the framing per
# NAL unit instead of reading it from the stream's avcC/hvcC extradata.
AVCC_VIDEOS = {
    "h264_in_mp4": (DALI_EXTRA_PATH + "/db/video/frame_num_timestamp/test_25fps.mp4", 250),
    "h264_in_mov": (DALI_EXTRA_PATH + "/db/video/containers/mov/cfr.mov", 180),
    "hevc_in_mp4": (DALI_EXTRA_PATH + "/db/video/hevc/sintel_trailer-720p.mp4", 1253),
}


@params(*AVCC_VIDEOS.keys())
def test_avcc_seek_matches_sequential_decode(case):
    """MP4/MOV store H.264/HEVC with AVCC (length-prefixed) NAL unit framing. This verifies that,
    for such content, a frame decoded after a seek is pixel-identical to the same frame decoded
    sequentially. Note: this is not a regression guard for the NAL-framing/keyframe-detection
    collision bug fixed alongside this test -- a missed keyframe there degrades seeking to a
    slower-but-still-pixel-correct fallback, so this test would likely have passed on the old,
    buggy parser too for these fixtures. The real regression guards for that bug are the C++
    NalFramingTest unit tests in frames_decoder_test.cc."""
    path, expected_frames = AVCC_VIDEOS[case]
    _check_seek_matches_sequential_decode(case, path, expected_frames)


def _check_seek_matches_sequential_decode(case, path, expected_frames):
    # Forward seek into the middle, backward seek, next frame, back to the start, last frame.
    targets = [
        expected_frames * 5 // 6,
        expected_frames // 2,
        expected_frames // 2 + 1,
        0,
        expected_frames - 1,
    ]

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def sequential_pipe():
        return fn.experimental.readers.video(
            device="gpu", filenames=[path], sequence_length=1, name="r"
        )

    try:
        p = sequential_pipe()
        p.build()
    except RuntimeError as e:
        raise AssertionError(
            f"[{case}] experimental.readers.video could not open/index {path}: {e}"
        )
    epoch_size = p.reader_meta("r")["epoch_size"]
    assert epoch_size == expected_frames, (
        f"[{case}] indexed {epoch_size} frames, expected {expected_frames}: the frame index "
        f"is wrong"
    )
    reference = {}
    for idx in range(epoch_size):
        (video,) = p.run()
        if idx in targets:
            reference[idx] = np.array(video.as_cpu()[0])[0]

    list_file = _write_file_list([f"{path} 0 {t} {t + 1}" for t in targets])

    @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
    def seeking_pipe():
        video, _, frame_num = fn.experimental.readers.video(
            device="gpu",
            file_list=list_file,
            file_list_format="frames",
            sequence_length=1,
            enable_frame_num=True,
        )
        return video, frame_num

    p = seeking_pipe()
    p.build()
    for t in targets:
        video, frame_num = p.run()
        got = int(np.array(frame_num.as_cpu()[0]).flatten()[0])
        assert got == t, f"[{case}] expected frame {t} next, reader returned frame {got}"
        frame = np.array(video.as_cpu()[0])[0]
        diff = np.abs(np.float32(frame) - np.float32(reference[t]))
        bad_fraction = np.count_nonzero(diff > 2) / diff.size
        assert bad_fraction <= 0.03, (
            f"[{case}] frame {t} decoded after a seek differs from the sequentially decoded "
            f"frame in {bad_fraction * 100:.1f}% of pixels: the seek landed on a wrong or "
            f"non-key frame (keyframe detection in FramesDecoderBase::BuildIndex)"
        )


@cartesian_params(["legacy", "experimental_cpu", "experimental_gpu"], ["absolute", "relative"])
def test_filename_with_colon_can_be_opened(reader, path_kind):
    """Detection test: experimental.readers.video opens files with
    avformat_open_input(filename), which may parse `clip:01.mp4` as a URL with protocol `clip`.
    Correct behavior: the file opens and all 50 frames are indexed, like any other file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        abs_name = os.path.join(tmp_dir, "clip:01.mp4")
        shutil.copy(CFR_VP9_60FPS_FILE, abs_name)
        name = abs_name if path_kind == "absolute" else "clip:01.mp4"
        if reader == "legacy":
            reader_fn = functools.partial(fn.readers.video, device="gpu")
        else:
            reader_fn = functools.partial(
                fn.experimental.readers.video, device=reader.split("_")[1]
            )

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
                raise AssertionError(f"[{reader}, {path_kind} path] could not open {name!r}: {e}")
            epoch_size = p.reader_meta("r")["epoch_size"]
            assert epoch_size == 50, (
                f"[{reader}, {path_kind} path] indexed {epoch_size} frames of {name!r}, "
                f"expected 50"
            )
            p.run()
        finally:
            os.chdir(prev_cwd)
