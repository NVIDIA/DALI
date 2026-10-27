# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cv2
import av
import io
import shutil
import struct
import subprocess
from fractions import Fraction
import tempfile
import functools
import nvidia.dali.types as types
import glob
import os
import random
from enum import Enum
from functools import lru_cache
from itertools import cycle
from test_utils import (
    get_dali_extra_path,
    is_mulit_gpu,
    skip_if_m60,
    compare_pipelines,
    check_output_pattern,
)
from nose2.tools import cartesian_params, params
from nose_utils import SkipTest, attr, assert_raises

filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
# filter out HEVC because some GPUs do not support it
filenames = filter(lambda filename: "hevc" not in filename, filenames)
# mpeg4 is not yet supported in the CPU operator itself
filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
filenames = filter(lambda filename: "av1" not in filename, filenames)

files = [np.fromfile(filename, dtype=np.uint8) for filename in filenames]


cfr_files = [
    f"{get_dali_extra_path()}/db/video/cfr/test_1_vp9.mp4",
    f"{get_dali_extra_path()}/db/video/cfr/test_2_vp9.mp4",
]
vfr_files = [
    f"{get_dali_extra_path()}/db/video/vfr/test_1_vp9.mp4",
    f"{get_dali_extra_path()}/db/video/vfr/test_2_vp9.mp4",
]

codec_files = {
    "h264": [f"{get_dali_extra_path()}/db/video/cfr/test_1.mp4"],
    "hevc": [f"{get_dali_extra_path()}/db/video/cfr/test_1_hevc.mp4"],
    "mpeg4": [f"{get_dali_extra_path()}/db/video/cfr/test_1_mpeg4.mp4"],
    "av1": [f"{get_dali_extra_path()}/db/video/cfr/test_1_av1.mp4"],
    "vp8": [f"{get_dali_extra_path()}/db/video/vp8/vp8.webm"],
    "vp9": [f"{get_dali_extra_path()}/db/video/vp9/vp9_0.mp4"],
}


# list all not supported codecs by the CPU operator we test on
unsupported_cpu_codec_error = r"is not supported by the CPU variant of this operator\."
unsupported_cpu_codecs = {"h264", "hevc", "mpeg4", "av1"}


def make_mjpeg_video(width, height):
    y, x = np.mgrid[:height, :width]
    x = x * 64 // max(width - 1, 1)
    y = y * 48 // max(height - 1, 1)
    encoded = io.BytesIO()
    with av.open(encoded, mode="w", format="avi") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width, stream.height = width, height
        # Avoid chroma upsampling differences between DALI's and PyAV's FFmpeg builds.
        stream.pix_fmt = "yuvj444p"
        for frame_idx in range(2):
            # Vary each channel along both axes so incorrect row/plane strides change pixels.
            pixels = np.stack(
                (
                    32 + x + y // 2 + 32 * frame_idx,
                    64 + x // 2 + y,
                    160 - x // 2 + y // 2 - 16 * frame_idx,
                ),
                axis=-1,
            ).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(
                pixels,
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return bytearray(encoded.getvalue())


def mjpeg_reference(data, image_type=types.RGB):
    with av.open(io.BytesIO(data)) as container:
        frames = []
        for frame in container.decode():
            if image_type == types.RGB:
                frames.append(frame.to_ndarray(format="rgb24"))
            else:
                # The CPU decoder produces full-range YCbCr in interleaved HWC layout.
                frames.append(frame.to_ndarray(format="yuvj444p").transpose(1, 2, 0))
        return np.stack(frames)


@params((32, 48), (30, 48), (64, 24), (64, 48))
def test_video_container_frame_dimensions(declared_width, declared_height):
    """Reject a container size that differs from the decoded frame size."""
    data = make_mjpeg_video(64, 48)
    # AVI's stream-format chunk contains a BITMAPINFOHEADER. Change its dimensions,
    # leaving the dimensions encoded in the MJPEG frames intact.
    header = data.index(b"strf")
    assert struct.unpack_from("<ii", data, header + 12) == (64, 48)
    struct.pack_into("<ii", data, header + 12, declared_width, declared_height)
    sample = np.frombuffer(data, dtype=np.uint8)

    @pipeline_def(batch_size=1, num_threads=1, device_id=None)
    def pipeline():
        video = fn.external_source(source=lambda: [sample], dtype=types.UINT8)
        return fn.decoders.video(video, device="cpu")

    pipe = pipeline()
    if (declared_width, declared_height) != (64, 48):
        with assert_raises(RuntimeError, glob="*Decoded frame dimensions (64x48)*"):
            pipe.run()
    else:
        np.testing.assert_allclose(pipe.run()[0].at(0), mjpeg_reference(data), atol=2, rtol=0)


@cartesian_params([30, 64], [types.RGB, types.YCbCr])
def test_video_reader_matching_frame_dimensions(width, image_type):
    """Exercise direct, padded RGB, and planar YCbCr conversion with valid frames."""
    data = make_mjpeg_video(width, 48)
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "video.avi")
        with open(filename, "wb") as video:
            video.write(data)

        @pipeline_def(batch_size=1, num_threads=1, device_id=None)
        def pipeline():
            return fn.experimental.readers.video(
                filenames=[filename], sequence_length=2, device="cpu", image_type=image_type
            )

        output = pipeline().run()[0].at(0)
        np.testing.assert_allclose(output, mjpeg_reference(data, image_type), atol=2, rtol=0)


def make_video(width, height, codec="libx264", num_frames=8, container_format="mp4", shift=0):
    """Encodes a moving gradient to an in-memory video. The stream crops the size to a coded one.

    A nonzero ``shift`` moves the timestamps back by that many frames, so that the first packets
    have negative ones (the muxer is told to keep them).
    """
    y, x = np.mgrid[:height, :width]
    encoded = io.BytesIO()
    options = {"avoid_negative_ts": "disabled"} if shift else {}
    with av.open(encoded, mode="w", format=container_format, options=options) as container:
        try:
            stream = container.add_stream(codec, rate=24)
        except Exception:
            raise SkipTest(f"{codec} is not available")
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = Fraction(1, 24)
        if codec == "libx265":
            stream.options = {"x265-params": "keyint=4:bframes=0:log-level=none", "crf": "10"}
        else:
            stream.options = {"g": "4", "bf": "0", "crf": "10"}
        packets = []
        for i in range(num_frames):
            pixels = np.stack(
                ((x * 255 // width + 8 * i) % 256, y * 255 // height, (x + y + 20 * i) % 256),
                axis=-1,
            ).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            packets.extend(stream.encode(frame))
        packets.extend(stream.encode())
        for packet in packets:
            if shift:
                packet.pts -= shift
                packet.dts -= shift
            container.mux(packet)
    return bytearray(encoded.getvalue())


def make_h264_video(width, height, num_frames=8):
    return make_video(width, height, "libx264", num_frames)


def set_mp4_container_size(data, width, height):
    """Rewrites the size in the sample entry. The size in the SPS is not changed."""
    stsd = data.index(b"stsd")
    entry = min(i for i in (data.find(tag, stsd) for tag in (b"avc1", b"hvc1", b"hev1")) if i > 0)
    struct.pack_into(">HH", data, entry + 28, width, height)


def recrop_with_ffmpeg(data, crop_options):
    """Rewrites the crop in the SPS of an H.264 stream, leaving the container size as it is."""
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SkipTest("the ffmpeg executable is not available")
    src = write_video(data, ".mp4")
    dst = src + ".crop.mp4"
    try:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-c", "copy"]
        subprocess.run(command + ["-bsf:v", f"h264_metadata={crop_options}", dst], check=True)
        with open(dst, "rb") as video:
            return bytearray(video.read())
    finally:
        for path in (src, dst):
            if os.path.exists(path):
                os.remove(path)


def decode_with_pyav(path):
    with av.open(path) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def read_video_gpu(path, num_frames, reader):
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def pipeline():
        if reader == "decoder":
            data = fn.external_source(
                source=lambda: [np.fromfile(path, dtype=np.uint8)], dtype=types.UINT8
            )
            return fn.decoders.video(data, device="mixed")
        if reader == "input":
            return fn.experimental.inputs.video(
                name="video_input",
                device="mixed",
                blocking=False,
                sequence_length=num_frames,
            )
        return fn.experimental.readers.video(
            filenames=[path], sequence_length=num_frames, device="gpu", random_shuffle=False
        )

    pipe = pipeline()
    if reader == "input":
        pipe.feed_input("video_input", [np.fromfile(path, dtype=np.uint8)])
    return np.array(pipe.run()[0].as_cpu().at(0))


def write_video(data, suffix):
    handle, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(handle, "wb") as video:
        video.write(data)
    return path


READERS = ("reader", "decoder", "input")


@params(
    ("libx264", 300, 232, (320, 240)),  # right and bottom
    ("libx264", 320, 232, (320, 240)),  # bottom only
    ("libx264", 300, 240, (320, 240)),  # right only
    ("libx264", 304, 240, (320, 256)),  # no crop in the stream, only the container is bigger
    ("libx264", 320, 240, (304, 232)),  # the container is smaller than the stream
    ("libx265", 300, 232, (320, 240)),
    ("libx265", 304, 240, (320, 256)),
)
def test_video_container_size_differs_from_stream_size(codec, width, height, container_size):
    """The output has the size of the picture in the stream, not the one from the container."""
    skip_if_m60()
    data = make_video(width, height, codec)
    set_mp4_container_size(data, *container_size)
    path = write_video(data, ".mp4")
    try:
        reference = decode_with_pyav(path)
        assert reference.shape[1:] == (height, width, 3), reference.shape
        for reader in READERS:
            output = read_video_gpu(path, len(reference), reader)
            assert output.shape == reference.shape, (reader, output.shape, reference.shape)
            diff = np.abs(output.astype(int) - reference.astype(int)).mean()
            assert diff < 2.0, (reader, diff)
    finally:
        os.remove(path)


@params("reader", "decoder")
def test_video_probe_keeps_leading_frames(reader):
    """Guards that a file with negative timestamps in the first packets (the seek to 0 skips them)
    returns the same frames as the sequential decoding, once the probe has rewound the file.

    The decoder builds the index of the file, which reopens it and so hides a wrong rewind. The
    reader only gives its decoder the index, so it decodes from where the probe left the file."""
    skip_if_m60()
    data = make_video(64, 64, num_frames=48, container_format="mpegts", shift=24)
    path = write_video(data, ".ts")
    try:
        with av.open(path) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 48 and sum(f.pts >= 0 for f in frames) == 24
        # the frames with negative timestamps are not indexed by DALI
        reference = np.stack([f.to_ndarray(format="rgb24") for f in frames if f.pts >= 0])
        output = read_video_gpu(path, len(reference), reader)
        assert output.shape == reference.shape, (output.shape, reference.shape)
        diff = np.abs(output.astype(int) - reference.astype(int)).mean(axis=(1, 2, 3))
        assert diff.max() < 3.5, diff
    finally:
        os.remove(path)


def make_video_without_leading_timestamps():
    return make_video(64, 64, num_frames=48, container_format="matroska", shift=24)


def test_video_no_timestamps_in_first_packets_decoder():
    """A file whose first packets have no timestamps is rejected by the index, as without the
    probe. The seek after the probe lands after those packets and must not hide them.
    Update the test if the index starts to accept such files."""
    skip_if_m60()
    path = write_video(make_video_without_leading_timestamps(), ".mkv")
    try:
        for reader in ("reader", "decoder"):
            with assert_raises(RuntimeError, regex="has no valid timestamps"):
                read_video_gpu(path, 48, reader)
    finally:
        os.remove(path)


def test_video_no_timestamps_in_first_packets_input():
    """The input operator does not index the file: all the frames are returned, as without the
    probe (and not the ones after the packets that the seek skipped)."""
    skip_if_m60()
    path = write_video(make_video_without_leading_timestamps(), ".mkv")
    try:
        reference = decode_with_pyav(path)
        output = read_video_gpu(path, len(reference), "input")
        assert output.shape == reference.shape, (output.shape, reference.shape)
        assert np.abs(output[0].astype(int) - reference[0].astype(int)).mean() < 2.0
    finally:
        os.remove(path)


@params(
    # a pure left/top crop is ignored by NVDEC: the full frame is output
    ("crop_left=64:crop_top=8", (240, 320)),
    # of a mixed crop, the right and bottom one is applied and the left and top one is not
    ("crop_left=16:crop_right=16:crop_top=8:crop_bottom=8", (232, 304)),
)
def test_video_left_top_crop_is_not_supported(crop_options, expected_size):
    """Documents the limitation: NVDEC does not report the left and top crop of the stream."""
    skip_if_m60()
    data = recrop_with_ffmpeg(make_h264_video(320, 240), crop_options)
    path = write_video(data, ".mp4")
    try:
        for reader in READERS:
            output = read_video_gpu(path, 8, reader)
            assert output.shape[1:3] == expected_size, (reader, output.shape)
    finally:
        os.remove(path)


def assert_unsupported_cpu_codec(run_pipeline):
    with check_output_pattern(unsupported_cpu_codec_error):
        with assert_raises(RuntimeError):
            run_pipeline()


def idx_reflect_101(idx, lo, hi):
    """Reflects out-of-range indices until fits in range.

    Args:
        idx: The index to reflect
        lo: Low (inclusive) bound
        hi: High (exclusive) bound

    Returns:
        Reflected index value
    """
    if hi - lo < 2:
        return hi - 1  # make it obviously wrong if hi <= lo

    while True:
        if idx < lo:
            idx = 2 * lo - idx
        elif idx >= hi:
            idx = 2 * hi - 2 - idx
        else:
            break

    return idx


def idx_reflect_1001(idx, lo, hi):
    """Reflects out-of-range indices until fits in range.

    Args:
        idx: The index to reflect
        lo: Low (inclusive) bound
        hi: High (exclusive) bound

    Returns:
        Reflected index value

    This reflect flavor repeats the first and last element:

    lo--v     v--hi
         ABCDEF           is padded as
    DCBAABCDEFFEDCBAABCD
    """
    if hi - lo < 1:
        return hi - 1  # make it obviously wrong if hi <= lo

    while True:
        if idx < lo:
            idx = 2 * lo - 1 - idx
        elif idx >= hi:
            idx = 2 * hi - 1 - idx
        else:
            break

    return idx


def constant_padding_frame(H, W, C, fill_value):
    return np.full((H, W, C), fill_value, dtype=np.uint8)


def reference_padded_video(full_video, frames, pad_mode, fill_value):
    F, H, W, C = full_video.shape

    # Generate indices for start:end:stride slice
    frame_indices = []
    for f in frames:
        if f < F:
            frame_indices.append(f)
        else:
            if pad_mode == "edge":
                frame_indices.append(F - 1)  # Repeat last frame
            elif pad_mode == "symmetric":
                # Reflect including boundary: abcd -> abccba
                frame_indices.append(idx_reflect_1001(f, 0, F))
            elif pad_mode == "reflect":
                # Reflect excluding boundary: abcd -> abcdcb
                frame_indices.append(idx_reflect_101(f, 0, F))
            elif pad_mode == "constant":
                frame_indices.append(-1)  # Special index for constant padding
            else:
                raise ValueError(f"Invalid pad_mode: {pad_mode}")

    out_reference = np.stack(
        [
            full_video[f] if f >= 0 else constant_padding_frame(H, W, C, fill_value)
            for f in frame_indices
        ]
    )
    return out_reference


def compare_videos(reference, actual):
    try:
        np.testing.assert_array_equal(reference, actual)
    except AssertionError:
        # Save frames as PNG files for debugging
        print("Videos don't match - saving frames as PNG files for debugging...")
        for i, (ref_frame, act_frame) in enumerate(zip(reference, actual)):
            ref_filename = f"reference_frame_{i}.png"
            act_filename = f"actual_frame_{i}.png"
            cv2.imwrite(ref_filename, ref_frame)
            cv2.imwrite(act_filename, act_frame)
            print(f"Saved frame {i} to {ref_filename} and {act_filename}")
        raise


@pipeline_def(device_id=0)
def video_decoder_pipeline(source, device="cpu", module=fn.experimental):
    data = fn.external_source(source=source, dtype=types.UINT8, ndim=1)
    return module.decoders.video(data, device=device)


def video_length(filename):
    cap = cv2.VideoCapture(filename)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def reference_pipeline(filename, device="cpu"):
    seq_length = video_length(filename)
    return fn.experimental.readers.video(
        filenames=[filename], sequence_length=seq_length, device=device
    )


def video_loader(batch_size, epochs):
    idx = 0
    while idx < epochs * len(files):
        batch = []
        for _ in range(batch_size):
            batch.append(files[idx % len(files)])
            idx = idx + 1
        yield batch


def video_decoder_iter(batch_size, epochs=1, device="cpu", module=fn.experimental):
    pipe = video_decoder_pipeline(
        batch_size=batch_size,
        device_id=0,
        num_threads=4,
        source=video_loader(batch_size, epochs),
        device=device,
        module=module,
    )
    for _ in range(int((epochs * len(files) + batch_size - 1) / batch_size)):
        (output,) = pipe.run()
        output = output.as_cpu()
        for i in range(batch_size):
            yield np.array(output[i])


def ref_iter(epochs=1, device="cpu"):
    for _ in range(epochs):
        for filename in filenames:
            pipe = reference_pipeline(filename, device=device)
            (output,) = pipe.run()
            output = output.as_cpu()
            yield np.array(output[0])


@params(("mixed", fn.experimental), ("mixed", fn))
def test_video_decoder(device, module):
    skip_if_m60()
    batch_size = 3
    epochs = 3
    decoder_iter = video_decoder_iter(batch_size, epochs, device, module=module)
    ref_dec_iter = ref_iter(epochs, device="cpu" if device == "cpu" else "gpu")
    for seq, ref_seq in zip(decoder_iter, ref_dec_iter):
        assert seq.shape == ref_seq.shape
        assert np.array_equal(seq, ref_seq)


@params(("video.mp4", "0001.png"), ("video_vp9.mp4", "0001_vp9.png"))
def test_full_range_video(filename, reference):
    skip_if_m60()

    @pipeline_def
    def test_pipeline():
        videos = fn.readers.video(
            device="gpu",
            filenames=[get_dali_extra_path() + f"/db/video/full_dynamic_range/{filename}"],
            sequence_length=1,
            initial_fill=10,
            normalized=False,
            dtype=types.UINT8,
        )
        return videos

    video_pipeline = test_pipeline(batch_size=1, num_threads=1, device_id=0)

    o = video_pipeline.run()
    out = o[0].as_cpu().as_array()
    ref = cv2.imread(get_dali_extra_path() + f"/db/video/full_dynamic_range/{reference}")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    left = ref
    right = out
    absdiff = np.abs(left.astype(int) - right.astype(int))
    assert np.mean(absdiff) < 2


@params(
    ("gpu", ("video.mp4", "0001.png")),
    ("gpu", ("video_vp9.mp4", "0001_vp9.png")),
    ("cpu", ("video_vp9.mp4", "0001_vp9.png")),
)
def test_full_range_video_experimental(device, video):
    skip_if_m60()
    filename, reference = video

    @pipeline_def
    def test_pipeline():
        videos = fn.experimental.readers.video(
            device=device,
            filenames=[get_dali_extra_path() + f"/db/video/full_dynamic_range/{filename}"],
            sequence_length=1,
        )
        return videos

    device_id = None if device == "cpu" else 0
    video_pipeline = test_pipeline(batch_size=1, num_threads=1, device_id=device_id)

    o = video_pipeline.run()
    out = o[0]
    if device == "gpu":
        out = out.as_cpu()
    out = out.as_array()
    ref = cv2.imread(get_dali_extra_path() + f"/db/video/full_dynamic_range/{reference}")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    left = ref
    right = out
    absdiff = np.abs(left.astype(int) - right.astype(int))
    assert np.mean(absdiff) < 2


COLOR_MATRIX_NUM_BARS = 10
COLOR_MATRIX_BAR_WIDTH = 192  # pixels


def color_matrix_bar(frame, i):
    """Mean color of the interior of the i-th bar of the test pattern in db/video/color_matrix: 10
    flat bars (red, green, blue, yellow, cyan, magenta, orange, grey, black, white), 192 pixels
    wide"""
    interior = (
        slice(100, 440),
        slice(i * COLOR_MATRIX_BAR_WIDTH + 32, (i + 1) * COLOR_MATRIX_BAR_WIDTH - 32),
    )
    return frame[interior].reshape(-1, 3).mean(axis=0)


def color_matrix_bars_error(frame, reference):
    """Worst difference between `frame` and `reference` of the mean colors of the interiors of the
    bars of the test pattern in db/video/color_matrix"""
    return max(
        np.max(np.abs(color_matrix_bar(frame, i) - color_matrix_bar(reference, i)))
        for i in range(COLOR_MATRIX_NUM_BARS)
    )


def check_color_matrix_black_and_white(frame):
    """Black and white are the ends of the range (Y=16 and Y=235 in the limited range): the luma
    has to be mapped to the whole range and clamped there (not 16 and 235). They do not depend on
    the matrix, unlike the saturated bars."""
    black, white = color_matrix_bar(frame, 8), color_matrix_bar(frame, 9)
    assert np.all(black <= 2), f"black bar is {black}"
    assert np.all(white >= 253), f"white bar is {white}"


def color_matrix_data_dir():
    return os.path.join(get_dali_extra_path(), "db", "video", "color_matrix")


def check_color_matrix_frame(frame, name):
    """Checks a decoded frame of the clip `name` of db/video/color_matrix against ffmpeg's decode"""
    ref_path = os.path.join(color_matrix_data_dir(), f"{name}_ref.png")
    ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    assert frame.shape == ref.shape
    assert color_matrix_bars_error(frame, ref) <= 4
    check_color_matrix_black_and_white(frame)
    assert np.mean(np.abs(frame.astype(int) - ref.astype(int))) < 2


@params("bt709_limited_1080p_vp9")
def test_color_matrix_bt709_cpu(name):
    # The streams are tagged as BT.709 (limited range). The references are what ffmpeg decodes,
    # which uses the matrix from the stream; converting with BT.601 gives errors up to 22 in the
    # interior of the saturated bars.

    @pipeline_def
    def test_pipeline():
        return fn.experimental.readers.video(
            device="cpu",
            filenames=[os.path.join(color_matrix_data_dir(), f"{name}.mp4")],
            sequence_length=1,
        )

    out = test_pipeline(batch_size=1, num_threads=1, device_id=None).run()[0]
    check_color_matrix_frame(out.as_array()[0, 0], name)


@params(
    "bt709_limited_1080p",
    "bt709_limited_1080p_vp9",
    "bt709_full_1080p",
    # the matrix is only in the H.264 stream, not in the container
    "bt709_limited_1080p_nocolr",
)
def test_color_matrix_bt709_gpu(name):
    # The streams are tagged as BT.709. The references are what ffmpeg decodes, which uses the
    # matrix and range from the stream; converting with BT.601 gives errors up to 22 in the
    # interior of the saturated bars.
    skip_if_m60()

    @pipeline_def
    def test_pipeline():
        return fn.experimental.readers.video(
            device="gpu",
            filenames=[os.path.join(color_matrix_data_dir(), f"{name}.mp4")],
            sequence_length=1,
        )

    out = test_pipeline(batch_size=1, num_threads=1, device_id=0).run()[0]
    out = out.as_cpu()
    check_color_matrix_frame(out.as_array()[0, 0], name)


def _ebml_element(element_id, payload):
    """An EBML element, with the data size in its 8 byte form"""
    return element_id + b"\x01" + len(payload).to_bytes(7, "big") + payload


def _ebml_uint(element_id, value):
    return _ebml_element(element_id, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))


# The timestamp of a block is a signed 16 bit offset (in ms) from the timestamp of its cluster
MATROSKA_MAX_CLUSTER_OFFSET_MS = 2**15


def write_matroska_without_color_and_frame_count(src, dst):
    """Remuxes the H.264 stream of `src` to a Matroska file, without re-encoding it. The file has
    the dimensions of the video, but neither color tags nor the number of frames (like many
    Matroska files): the only color description is in the H.264 stream itself. The container is
    written here because of the lack of the tools that would leave out the color tags."""
    with av.open(src) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        extradata = bytes(codec.extradata)
        width, height = codec.width, codec.height
        frame_duration_ns = int(1e9 / stream.average_rate) if stream.average_rate else None
        # (timestamp in ms, is keyframe, data), in the decoding order
        frames = [
            (int(round(float(p.pts * stream.time_base) * 1000)), p.is_keyframe, bytes(p))
            for p in container.demux(stream)
            if p.pts is not None and p.dts is not None
        ]

    ebml_header = _ebml_element(
        b"\x1a\x45\xdf\xa3",
        _ebml_uint(b"\x42\x86", 1)  # EBMLVersion
        + _ebml_uint(b"\x42\xf7", 1)  # EBMLReadVersion
        + _ebml_uint(b"\x42\xf2", 4)  # EBMLMaxIDLength
        + _ebml_uint(b"\x42\xf3", 8)  # EBMLMaxSizeLength
        + _ebml_element(b"\x42\x82", b"matroska")  # DocType
        + _ebml_uint(b"\x42\x87", 4)  # DocTypeVersion
        + _ebml_uint(b"\x42\x85", 2),  # DocTypeReadVersion
    )
    info = _ebml_element(
        b"\x15\x49\xa9\x66", _ebml_uint(b"\x2a\xd7\xb1", 1000000)  # TimecodeScale: 1 ms
    )
    track = _ebml_element(
        b"\xae",  # TrackEntry
        _ebml_uint(b"\xd7", 1)  # TrackNumber
        + _ebml_uint(b"\x73\xc5", 1)  # TrackUID
        + _ebml_uint(b"\x83", 1)  # TrackType: video
        + (
            _ebml_uint(b"\x23\xe3\x83", frame_duration_ns)  # DefaultDuration
            if frame_duration_ns
            else b""
        )
        + _ebml_element(b"\x86", b"V_MPEG4/ISO/AVC")  # CodecID
        + _ebml_element(b"\x63\xa2", extradata)  # CodecPrivate: avcC
        + _ebml_element(
            b"\xe0",  # Video, without the Colour element
            _ebml_uint(b"\xb0", width) + _ebml_uint(b"\xba", height),
        ),
    )
    tracks = _ebml_element(b"\x16\x54\xae\x6b", track)
    # (timestamp of the cluster in ms, its blocks): a new cluster when the offset doesn't fit
    clusters = []
    for ts, key, data in frames:
        if not clusters or abs(ts - clusters[-1][0]) >= MATROSKA_MAX_CLUSTER_OFFSET_MS:
            clusters.append((ts, []))
        cluster_ts, blocks = clusters[-1]
        blocks.append(
            _ebml_element(
                b"\xa3",  # SimpleBlock: track 1, timestamp relative to the cluster, flags
                b"\x81"
                + (ts - cluster_ts).to_bytes(2, "big", signed=True)
                + (b"\x80" if key else b"\x00")
                + data,
            )
        )
    cluster_elements = b"".join(
        _ebml_element(b"\x1f\x43\xb6\x75", _ebml_uint(b"\xe7", cluster_ts) + b"".join(blocks))
        for cluster_ts, blocks in clusters
    )
    segment = _ebml_element(b"\x18\x53\x80\x67", info + tracks + cluster_elements)
    with open(dst, "wb") as f:
        f.write(ebml_header + segment)


def test_color_matrix_bt709_gpu_matroska_without_color_tags():
    # Matroska files often have neither color tags nor the number of frames. The matrix of this
    # clip is then only in the H.264 stream (SPS), and the decoder has to take it from there. As
    # there is no number of frames, the decoder also reads the stream to count the frames and
    # reopens the file, so this also checks that the matrix doesn't depend on what the decoder
    # found when it opened the file the first time.
    skip_if_m60()
    name = "bt709_limited_1080p_nocolr"
    with tempfile.TemporaryDirectory() as tmp_dir:
        mkv_path = os.path.join(tmp_dir, f"{name}.mkv")
        write_matroska_without_color_and_frame_count(
            os.path.join(color_matrix_data_dir(), f"{name}.mp4"), mkv_path
        )
        encoded = np.fromfile(mkv_path, dtype=np.uint8)

    @pipeline_def
    def test_pipeline():
        data = fn.external_source(source=lambda: [encoded], dtype=types.UINT8, ndim=1)
        return fn.decoders.video(data, device="mixed")

    out = test_pipeline(batch_size=1, num_threads=1, device_id=0).run()[0]
    out = out.as_cpu()
    check_color_matrix_frame(out.as_array()[0][0], name)


@attr("multi_gpu")
@params("cpu", "mixed")
def test_multi_gpu_video(device):
    skip_if_m60()
    if not is_mulit_gpu():
        raise SkipTest()

    batch_size = 1

    def input_gen(batch_size):
        filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
        # test overflow of frame_buffer_
        if device == "mixed":
            filenames.append(f"{get_dali_extra_path()}/db/video/cfr_test.mp4")
        filenames = filter(lambda filename: "hevc" not in filename, filenames)
        filenames = filter(lambda filename: "av1" not in filename, filenames)
        if device == "cpu":
            # some formats are not yet supported in the CPU operator itself
            filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
            excluded = {"test_1.mp4", "test_2.mp4"}
            filenames = filter(lambda f: os.path.basename(f) not in excluded, filenames)
        filenames = cycle(filenames)
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(np.fromfile(next(filenames), dtype=np.uint8))
            yield batch

    @pipeline_def
    def test_pipeline():
        vid = fn.external_source(device="cpu", source=input_gen(batch_size))
        seq = fn.experimental.decoders.video(vid, device=device)
        return seq

    video_pipeline_0 = test_pipeline(batch_size=1, num_threads=1, device_id=0)
    video_pipeline_1 = test_pipeline(batch_size=1, num_threads=1, device_id=1)

    iters = 5
    for _ in range(iters):
        video_pipeline_0.run()
        video_pipeline_1.run()


@params("cpu", "gpu")
def test_source_info(device):
    skip_if_m60()
    filenames = glob.glob(f"{get_dali_extra_path()}/db/video/[cv]fr/*.mp4")
    # filter out HEVC because some GPUs do not support it
    filenames = filter(lambda filename: "hevc" not in filename, filenames)
    # filter out AV1 because some GPUs do not support it
    filenames = filter(lambda filename: "av1" not in filename, filenames)
    if device == "cpu":
        # some formats are not yet supported in the CPU operator itself
        filenames = filter(lambda filename: "mpeg4" not in filename, filenames)
        excluded = {"test_1.mp4", "test_2.mp4"}
        filenames = filter(lambda f: os.path.basename(f) not in excluded, filenames)

    files = list(filenames)

    @pipeline_def
    def test_pipeline():
        videos = fn.experimental.readers.video(
            device=device,
            filenames=files,
            sequence_length=1,
            step=10000000,  # make sure that each video has only one valid sequence
        )
        return videos

    batch_size = 4
    device_id = None if device == "cpu" else 0
    p = test_pipeline(batch_size=batch_size, num_threads=1, device_id=device_id)

    samples_read = 0
    while samples_read < len(files):
        o = p.run()
        for idx, t in enumerate(o[0]):
            assert t.source_info() == files[(samples_read + idx) % len(files)]
        samples_read += batch_size


@params("cpu", "mixed")
def test_error_source_info(device):
    skip_if_m60()
    error_file = "README.txt"
    filenames = os.path.join(get_dali_extra_path(), "db/video/cfr/", error_file)

    @pipeline_def
    def test_pipeline():
        data, _ = fn.readers.file(files=filenames)
        return fn.experimental.decoders.video(data, device=device)

    batch_size = 4
    p = test_pipeline(batch_size=batch_size, num_threads=1, device_id=0)

    assert_raises(RuntimeError, p.run)


@params(
    # Test case 1: Constant frame rate video with sequence_length
    *[(device, 3, cfr_files, 5, 3, 1, None, True) for device in ["cpu", "mixed"]],
    # Test case 2: Variable frame rate video with sequence_length
    *[(device, 3, vfr_files, 0, 4, 2, None, True) for device in ["cpu", "mixed"]],
    # Test case 3: Constant frame rate video with sequence_length and padding
    *[
        (device, 3, cfr_files, 0, 10, 11, pad_mode, True)
        for device in ["cpu", "mixed"]
        for pad_mode in ["constant", "edge", "symmetric", "reflect"]
    ],
    # Test case 4: Constant frame rate video with end_frame
    *[(device, 3, cfr_files, 5, 8, 1, None, False) for device in ["cpu", "mixed"]],
    # Test case 5: Variable frame rate video with end_frame
    *[(device, 3, vfr_files, 0, 8, 2, None, False) for device in ["cpu", "mixed"]],
    # Test case 6: Constant frame rate video with end_frame and padding
    *[
        (device, 3, cfr_files, 0, 110, 11, pad_mode, False)
        for device in ["cpu", "mixed"]
        for pad_mode in ["constant", "edge", "symmetric", "reflect"]
    ],
    # Test case 7: Random parameters for both sequence_length and end_frame
    *[
        (device, 3, cfr_files, "random", "random", "random", None, use_seq_len)
        for device in ["cpu", "mixed"]
        for use_seq_len in [True, False]
    ],
)
def test_video_decoder_frame_start_end_stride(
    device, batch_size, filenames, start_frame, length_or_end, stride, pad_mode, use_sequence_length
):
    skip_if_m60()
    num_iters = 1
    batch = []
    fill_value = 111
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        reference = fn.experimental.decoders.video(encoded, device=device)

        start_frame_arg = (
            fn.random.uniform(range=(0, 5), dtype=types.INT32)
            if start_frame == "random"
            else start_frame
        )

        if length_or_end == "random":
            if use_sequence_length:
                length_or_end_arg = fn.random.uniform(range=(3, 6), dtype=types.INT32)
            else:
                length_or_end_arg = fn.random.uniform(range=(6, 10), dtype=types.INT32)
        else:
            length_or_end_arg = length_or_end

        stride_arg = (
            fn.random.uniform(range=(1, 4), dtype=types.INT32) if stride == "random" else stride
        )

        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            start_frame=start_frame_arg,
            sequence_length=length_or_end_arg if use_sequence_length else None,
            end_frame=None if use_sequence_length else length_or_end_arg,
            stride=stride_arg,
            pad_mode=pad_mode,
            fill_value=fill_value,
        )

        return (reference, decoded, start_frame_arg, length_or_end_arg, stride_arg)

    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=0)

    for _ in range(num_iters):
        out0, out1, start_frame_arg, length_or_end_arg, stride_arg = (
            o.as_cpu() for o in pipe.run()
        )
        # Verify each sample in the batch
        for i in range(batch_size):
            out_reference_full = out0.at(i)
            out = out1.at(i)
            start_frame = start_frame_arg.at(i)
            length_or_end = length_or_end_arg.at(i)
            stride = stride_arg.at(i)

            if use_sequence_length:
                frames = list(range(start_frame, start_frame + length_or_end * stride, stride))
            else:
                frames = list(range(start_frame, length_or_end, stride))
            out_reference = reference_padded_video(out_reference_full, frames, pad_mode, fill_value)

            compare_videos(out_reference, out)


@params(
    # Test single constant frame rate video with simple frame indices
    *[(device, 3, cfr_files, [0, 5, 10], "none") for device in ["cpu", "mixed"]],
    # Test multiple variable frame rate videos with non-sequential frames
    *[(device, 3, vfr_files, [2, 4, 8], "none") for device in ["cpu", "mixed"]],
    # Test single constant frame rate video with non-monotonic frame indices
    *[(device, 3, cfr_files, [0, 5, 10, 8, 7, 6], "none") for device in ["cpu", "mixed"]],
    # Test single constant frame rate video with repeated frame indices
    *[(device, 3, cfr_files, [0, 5, 10, 8, 10, 5, 0, 0], "none") for device in ["cpu", "mixed"]],
    # Test multiple constant frame rate videos with random indices
    *[(device, 3, cfr_files, None, "none") for device in ["cpu", "mixed"]],
    # Test out of bounds frame indices
    *[
        (device, 3, cfr_files, [0, 100, 1, 100, 2, 100, 3, 100], "constant")
        for device in ["cpu", "mixed"]
    ],
)
def test_video_decoder_frame_indices(device, batch_size, filenames, frames, pad_mode):
    skip_if_m60()
    num_iters = 1
    batch = []
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        return batch

    fill_value = [118, 185, 0]

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        reference = fn.experimental.decoders.video(encoded, device=device)
        if frames is None:
            frames_shape = fn.random.uniform(range=(1, 10), shape=(1,), dtype=types.INT32)
            frames_arg = fn.random.uniform(range=(0, 10), shape=frames_shape, dtype=types.INT32)
        else:
            frames_arg = frames
        decoded = fn.experimental.decoders.video(
            encoded, device=device, frames=frames_arg, pad_mode=pad_mode, fill_value=fill_value
        )
        return (reference, decoded, frames_arg)

    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    for _ in range(num_iters):
        out0, out1, frames_arg = (o.as_cpu() for o in pipe.run())
        # Verify each sample in the batch
        for i in range(batch_size):
            full_video = out0.at(i)
            F, H, W, C = full_video.shape
            out = out1.at(i)
            frames = frames_arg.at(i)
            # Get frames at specified indices for reference
            if pad_mode == "none":
                for f in frames:
                    assert f < full_video.shape[0]
                out_reference = full_video[frames]
            else:
                padding = np.full((H, W, C), fill_value, dtype=np.uint8)
                out_reference = np.zeros((len(frames), H, W, C), dtype=np.uint8)
                for f in range(len(frames)):
                    if frames[f] < F:
                        out_reference[f] = full_video[frames[f]]
                    else:
                        out_reference[f] = padding
            compare_videos(out_reference, out)


@params(
    # Test single constant frame rate video with start_frame near the end
    *[(device, 3, cfr_files, 1, 1000) for device in ["cpu", "mixed"]],
    # Test multiple variable frame rate videos with large stride
    *[(device, 3, vfr_files, 2, 1000) for device in ["cpu", "mixed"]],
)
def test_video_decoder_no_padding(device, batch_size, filenames, stride, sequence_length):
    skip_if_m60()
    batch = []
    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        # Get full video first as reference
        reference = fn.experimental.decoders.video(encoded, device=device, stride=stride)
        # Request frames that would exceed video length
        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            stride=stride,
            sequence_length=sequence_length,
            pad_mode="none",
        )
        return (reference, decoded)

    pipe = test_pipeline(batch_size=batch_size, num_threads=1, device_id=0)
    out_ref, out = (o.as_cpu() for o in pipe.run())

    # Verify each sample in the batch
    for i in range(batch_size):
        ref = out_ref.at(i)
        actual = out.at(i)
        compare_videos(ref, actual)


@params("cpu", "mixed")
def test_incompatible_args(device):
    skip_if_m60()
    batch = []
    with open(cfr_files[0], "rb") as f:
        batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline(
        frames=None, start_frame=None, stride=None, sequence_length=None, end_frame=None
    ):
        encoded = fn.external_source(source=get_batch, device="cpu")
        decoded = fn.experimental.decoders.video(
            encoded,
            device=device,
            frames=frames,
            start_frame=start_frame,
            end_frame=end_frame,
            sequence_length=sequence_length,
            stride=stride,
        )
        return decoded

    err_msg_0 = (
        "Cannot specify both `frames` and any of `start_frame`, "
        "`sequence_length`, `stride`, `end_frame` arguments"
    )
    err_msg_1 = "Cannot specify both `sequence_length` and `end_frame` arguments"

    # Test that frames argument is incompatible with start_frame, stride, sequence_length
    # and end_frame
    for frames, start_frame, stride, sequence_length, end_frame, err_msg in [
        # Test `frames` with `start_frame`
        ([0, 1, 2], 0, None, None, None, err_msg_0),
        # Test `frames` with `stride`
        ([0, 1, 2], None, 1, None, None, err_msg_0),
        # Test `frames` with `sequence_length`
        ([0, 1, 2], None, None, 3, None, err_msg_0),
        # Test `frames` with multiple incompatible args
        ([0, 1, 2], 0, 1, 3, None, err_msg_0),
        # Test `frames` with `end_frame`
        ([0, 1, 2], None, None, None, 5, err_msg_0),
        # Test `frames` with `start_frame` and `end_frame`
        ([0, 1, 2], 0, None, None, 5, err_msg_0),
        # Test `frames` with `stride` and `end_frame`
        ([0, 1, 2], None, 1, None, 5, err_msg_0),
        # Test `sequence_length` and `end_frame`
        (None, None, None, 10, 10, err_msg_1),
    ]:
        with assert_raises(
            RuntimeError,
            glob=err_msg,
        ):
            pipe = test_pipeline(
                batch_size=1,
                num_threads=3,
                device_id=0,
                frames=frames,
                start_frame=start_frame,
                stride=stride,
                sequence_length=sequence_length,
                end_frame=end_frame,
            )
            pipe.build()


@params("cpu", "mixed")
def test_multichannel_fill_value(device):
    skip_if_m60()
    batch = []
    batch_size = 3
    fill_value = [118, 185, 0]  # RGB fill values

    for i in range(batch_size):
        with open(cfr_files[i % len(cfr_files)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def test_pipeline():
        encoded = fn.external_source(source=get_batch, device="cpu")
        decoded0 = fn.experimental.decoders.video(
            encoded,
            stride=10,
            device=device,
            pad_mode="none",
        )
        decoded1 = fn.experimental.decoders.video(
            encoded,
            device=device,
            stride=10,
            sequence_length=10,  # Request more frames than available
            pad_mode="constant",
            fill_value=fill_value,
        )
        return decoded0, decoded1

    batch_size = 3
    device_id = None if device == "cpu" else 0
    pipe = test_pipeline(batch_size=batch_size, num_threads=3, device_id=device_id)
    out = pipe.run()
    out0, out1 = (o.as_cpu() for o in out)

    for i in range(batch_size):
        frames = out0.at(i)
        frames_padded = out1.at(i)

        padded_F = frames_padded.shape[0]
        F, H, W, C = frames.shape
        padding = padded_F - F
        assert padding > 0

        # Create expected padded frames with RGB fill values
        padded_frame = np.full((padding, H, W, C), fill_value, dtype=np.uint8)
        np.testing.assert_array_equal(frames_padded[:F, :, :, :], frames)
        np.testing.assert_array_equal(frames_padded[F:, :, :, :], padded_frame)


def extract_frames_from_video(
    video_path, start_frame=None, sequence_length=None, end_frame=None, stride=None
):
    """Extracts frames from a video file using PyAV.

    Args:
        video_path: Path to the video file
        start_frame: Starting frame index
        sequence_length: Number of frames to extract (cannot be used together with end_frame)
        end_frame: Ending frame index (exclusive) (cannot be used together with sequence_length)
        stride: Number of frames to skip between each extracted frame

    Returns:
        List of frames as numpy arrays
    """

    frames = []

    if sequence_length is not None and end_frame is not None:
        raise ValueError("Cannot specify both sequence_length and end_frame")

    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Calculate correct starting point
    start = start_frame or 0
    stride = stride or 1
    frames_read = 0
    decoded_frame_idx = 0

    for frame in container.decode(video_stream):
        # Early exit if possible
        if (sequence_length is not None and frames_read >= sequence_length) or (
            end_frame is not None and decoded_frame_idx >= end_frame
        ):
            break
        if decoded_frame_idx < start:
            decoded_frame_idx += 1
            continue

        # If stride requested, process only on stride-boundary
        if (decoded_frame_idx - start) % stride == 0:
            # SWS_BILINEAR(0x40)|SWS_FULL_CHR_H_INT(0x2000)|SWS_ACCURATE_RND(0x40000)
            nd_frame = frame.to_ndarray(format="rgb24", interpolation=0x40 + 0x2000 + 0x40000)
            frames.append(nd_frame)
            frames_read += 1

        decoded_frame_idx += 1

    container.close()
    return frames


def compare_frames(frame, ref_frame, frame_idx, diff_step=2, threshold=0.03):
    # Compare frames
    diff_pixels = np.count_nonzero(np.abs(np.float32(frame) - np.float32(ref_frame)) > diff_step)
    total_pixels = frame.size
    # More than 3% of the pixels differ in more than 2 steps
    if diff_pixels / total_pixels > threshold:
        # Save the mismatched frames for inspection
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ref_frame_bgr = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)

        output_path = f"frame_{frame_idx+1:03d}.png"
        ref_output_path = f"ref_frame_{frame_idx+1:03d}.png"

        cv2.imwrite(output_path, frame_bgr)
        cv2.imwrite(ref_output_path, ref_frame_bgr)
        assert False, (
            f"Frame {frame_idx+1} differs from reference by more than {diff_step} steps in "
            + f"{diff_pixels/total_pixels*100}% of pixels. "
            + f"Expected {ref_frame_bgr} but got {frame_bgr}"
        )


@params(
    *[
        (device, codec, start_frame, sequence_length, end_frame, stride)
        for device in ["cpu", "mixed"]
        for codec in ["h264", "hevc", "vp8", "vp9", "av1", "mpeg4"]
        for start_frame, sequence_length, end_frame, stride in [
            (None, None, None, None),  # Default case
            (5, 3, None, 2),  # start_frame, sequence_length and stride
            (5, None, 10, 2),  # start_frame, end_frame and stride
        ]
    ]
)
def test_decoder_operator_codec_support(
    device, codec, start_frame, sequence_length, end_frame, stride
):
    skip_if_m60()
    filenames = codec_files[codec]
    assert len(filenames) > 0, f"No {codec} test files found"

    if codec == "av1":
        raise SkipTest(f"Codec {codec} is only supported by Ampere+ GPUs, skipping test for now.")

    batch_size = 1
    diff_step = 5 if codec == "mpeg4" else 2

    batch = []

    def get_batch():
        random.shuffle(batch)
        return batch

    @pipeline_def
    def decoder_pipeline():
        files = fn.external_source(source=get_batch)
        videos = fn.experimental.decoders.video(
            files,
            device=device,
            start_frame=start_frame,
            end_frame=end_frame,
            sequence_length=sequence_length,
            stride=stride,
        )
        return videos

    device_id = None if device == "cpu" else 0
    pipe = decoder_pipeline(batch_size=batch_size, num_threads=2, device_id=device_id)
    if device == "cpu" and codec in unsupported_cpu_codecs:
        # Only the codec-detection path is exercised; load a single file so the decoder
        # has bytes to inspect, then return without consuming the full test batch.
        with open(filenames[0], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        assert_unsupported_cpu_codec(pipe.run)
        return

    for i in range(batch_size):
        with open(filenames[i % len(filenames)], "rb") as f:
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))

    (out,) = pipe.run()
    assert len(out) > 0, f"No output from decoder pipeline for {codec}"

    for i in range(batch_size):
        frames = out.as_cpu().at(i)
        assert frames.shape[0] > 0, f"No frames decoded for sample {i}"
        # Get the filename for the i-th sample
        filename = filenames[i % len(filenames)]
        reference_frames = extract_frames_from_video(
            filename, start_frame, sequence_length, end_frame, stride
        )
        for frame_idx, frame in enumerate(frames):
            compare_frames(
                frame, reference_frames[frame_idx], frame_idx, diff_step=diff_step, threshold=0.03
            )


@params(
    *[
        (device, codec)
        for device in ["cpu", "gpu"]
        for codec in ["h264", "hevc", "vp8", "vp9", "mpeg4", "av1"]
    ]
)
def test_reader_operator_codec_support(device, codec, sequence_length=3, stride=2):
    skip_if_m60()
    filenames = codec_files[codec]
    assert len(filenames) > 0, f"No {codec} test files found"

    if codec == "av1":
        raise SkipTest(f"Codec {codec} is only supported by Ampere+ GPUs, skipping test for now.")

    batch_size = 1
    diff_step = 5 if codec == "mpeg4" else 2

    @pipeline_def
    def decoder_pipeline():
        videos, frame_no = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num="scalar",
        )
        return videos, frame_no

    device_id = None if device == "cpu" else 0
    pipe = decoder_pipeline(batch_size=batch_size, num_threads=2, device_id=device_id)
    if device == "cpu" and codec in unsupported_cpu_codecs:
        assert_unsupported_cpu_codec(pipe.run)
        return

    out, frame_no = pipe.run()
    assert len(out) > 0, f"No output from decoder pipeline for {codec}"

    for i in range(batch_size):
        frames = out.as_cpu().at(i)
        frame_no_cpu = int(frame_no.as_cpu().at(i).item())
        assert frames.shape[0] > 0, f"No frames decoded for sample {i}"

        # Get the filename for the i-th sample
        filename = filenames[i % len(filenames)]
        reference_frames = extract_frames_from_video(
            filename, frame_no_cpu, sequence_length, None, stride
        )
        for frame_idx, frame in enumerate(frames):
            compare_frames(
                frame, reference_frames[frame_idx], frame_idx, diff_step=diff_step, threshold=0.03
            )


def test_no_first_key_frame():
    batch_size = 2
    sequence_length = 8
    initial_prefetch_size = 16
    data_root_dir = get_dali_extra_path()
    video_files = glob.glob(f"{data_root_dir}/db/video/sintel/labelled_videos/0/*.mp4")

    @pipeline_def
    def video_pipe(filenames):
        videos = fn.experimental.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            initial_fill=initial_prefetch_size,
        )
        return videos

    pipe = video_pipe(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        filenames=video_files,
        seed=123456,
    )

    pipe.run()


@cartesian_params(["cpu", "gpu"], [1, 2, 3])
def test_enable_frame_num_sequence_basic(device, stride):
    """Test that enable_frame_num='sequence' returns correct per-frame frame indices."""
    skip_if_m60()
    sequence_length = 5
    batch_size = 2
    filenames = cfr_files

    @pipeline_def
    def video_pipe():
        videos, frame_idxs = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num="sequence",
        )
        return videos, frame_idxs

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0)
    out_videos, out_frame_idxs = pipe.run()

    for i in range(batch_size):
        idxs = out_frame_idxs.as_cpu().at(i)

        # Output should have shape (sequence_length,)
        assert idxs.shape == (
            sequence_length,
        ), f"Expected shape ({sequence_length},), got {idxs.shape}"

        # All frames should be valid (>= 0) since we use default pad_mode='none'
        # and the video has enough frames
        assert all(
            idx >= 0 for idx in idxs.flatten()
        ), f"Expected all frame indices >= 0, got {idxs}"

        # Frame indices should be monotonically increasing with step=stride
        idxs_flat = idxs.flatten()
        for j in range(1, len(idxs_flat)):
            assert idxs_flat[j] == idxs_flat[j - 1] + stride, (
                f"Expected consecutive frame indices to differ by stride={stride}, "
                f"got {idxs_flat[j-1]} and {idxs_flat[j]}"
            )


@params(*[(device,) for device in ["cpu", "gpu"]])
def test_enable_frame_num_sequence_with_padding(device):
    """Test that enable_frame_num='sequence' returns -1 for constant-padded frames."""
    skip_if_m60()
    sequence_length = 8
    stride = 3
    batch_size = 1
    video_file = cfr_files[0]

    # Count total frames in the video
    container = av.open(video_file)
    video_stream = next(s for s in container.streams if s.type == "video")
    total_frames = video_stream.frames
    container.close()

    @pipeline_def
    def video_pipe():
        videos, frame_idxs = fn.experimental.readers.video(
            filenames=[video_file],
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            pad_mode="constant",
            enable_frame_num="sequence",
        )
        return videos, frame_idxs

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0)

    # Iterate until the tail (padded) sequence is reached
    full_sequences = total_frames // (stride * sequence_length)
    padded_idxs = None
    for _ in range(full_sequences + 2):
        _, out_idxs = pipe.run()
        idxs = out_idxs.as_cpu().at(0).flatten()
        if -1 in idxs:
            padded_idxs = idxs
            break

    assert (
        padded_idxs is not None
    ), "No padded sample encountered - expected a tail sequence with -1 indices"
    first_pad_pos = np.where(padded_idxs == -1)[0][0]
    assert all(
        padded_idxs[:first_pad_pos] >= 0
    ), f"Expected non-negative frame indices before padding, got {padded_idxs}"
    assert all(
        padded_idxs[first_pad_pos:] == -1
    ), f"Expected -1 for all padded frames, got {padded_idxs}"


def _ref_clamp(idx, size):
    return max(0, min(idx, size - 1))


def _ref_reflect_1001(idx, size):
    """idx_reflect_1001(idx, 0, size): symmetric reflection including endpoints."""
    if size <= 0:
        return 0
    period = 2 * size
    idx = idx % period
    if idx < 0:
        idx += period
    return idx if idx < size else 2 * size - 1 - idx


def _ref_reflect_101(idx, size):
    """idx_reflect_101(idx, 0, size): reflection without repeating endpoints."""
    if size <= 1:
        return 0
    period = 2 * (size - 1)
    idx = idx % period
    if idx < 0:
        idx += period
    return idx if idx < size else 2 * (size - 1) - idx


_BOUNDARY_REF = {
    "edge": _ref_clamp,
    "reflect_1001": _ref_reflect_1001,
    "reflect_101": _ref_reflect_101,
}


@cartesian_params(["cpu", "gpu"], ["edge", "reflect_1001", "reflect_101"])
def test_enable_frame_num_sequence_non_constant_padding(device, pad_mode):
    """Test enable_frame_num='sequence' with non-constant pad modes (edge, reflect).

    The experimental reader uses HandleBoundary() which returns reflected/clamped
    indices for these modes instead of -1 (which is only returned for pad_mode='constant').
    """
    skip_if_m60()
    sequence_length = 8
    stride = 2
    batch_size = 1
    video_file = cfr_files[0]

    container = av.open(video_file)
    video_stream = next(s for s in container.streams if s.type == "video")
    total_frames = video_stream.frames
    container.close()

    full_seq_stride = stride * sequence_length
    if total_frames % full_seq_stride == 0:
        raise SkipTest(
            f"Video has {total_frames} frames, exactly divisible by {full_seq_stride}; "
            f"no padding needed"
        )

    tail_start = (total_frames // full_seq_stride) * full_seq_stride
    ref_fn = _BOUNDARY_REF[pad_mode]
    expected = [ref_fn(tail_start + i * stride, total_frames) for i in range(sequence_length)]

    @pipeline_def
    def video_pipe():
        videos, frame_idxs = fn.experimental.readers.video(
            filenames=[video_file],
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            pad_mode=pad_mode,
            enable_frame_num="sequence",
        )
        return videos, frame_idxs

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0)

    num_full_seqs = total_frames // full_seq_stride
    padded_idxs = None
    for _ in range(num_full_seqs + 2):
        _, out_idxs = pipe.run()
        idxs = out_idxs.as_cpu().at(0).flatten()
        if idxs[0] == tail_start:
            padded_idxs = idxs
            break

    assert padded_idxs is not None, (
        f"No tail sequence starting at frame {tail_start} was found in "
        f"{num_full_seqs + 2} iterations"
    )
    assert all(
        idx >= 0 for idx in padded_idxs
    ), f"pad_mode='{pad_mode}' should not produce -1 frame indices, got {padded_idxs}"
    for j, (got, exp) in enumerate(zip(padded_idxs.tolist(), expected)):
        assert got == exp, (
            f"Frame {j} (source idx {tail_start + j * stride}): "
            f"got {got}, expected {exp} for pad_mode='{pad_mode}'"
        )


@params(*[(device,) for device in ["cpu", "gpu"]])
def test_enable_frame_num_sequence_matches_scalar(device):
    """Test that the first element of 'sequence' output matches 'scalar' output."""
    skip_if_m60()
    sequence_length = 5
    stride = 2
    batch_size = 2
    filenames = cfr_files

    @pipeline_def
    def scalar_pipe():
        videos, frame_idx = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num="scalar",
        )
        return videos, frame_idx

    @pipeline_def
    def sequence_pipe():
        videos, frame_idxs = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num="sequence",
        )
        return videos, frame_idxs

    pipe_s = scalar_pipe(batch_size=batch_size, num_threads=2, device_id=0, seed=42)
    pipe_q = sequence_pipe(batch_size=batch_size, num_threads=2, device_id=0, seed=42)
    _, out_scalar = pipe_s.run()
    _, out_sequence = pipe_q.run()

    for i in range(batch_size):
        scalar_idx = int(out_scalar.as_cpu().at(i)[0])
        seq_idxs = out_sequence.as_cpu().at(i).flatten()
        assert seq_idxs[0] == scalar_idx, (
            f"First element of 'sequence' output ({seq_idxs[0]}) should match "
            f"'scalar' output ({scalar_idx})"
        )


@params("cpu", "gpu")
def test_enable_frame_num_true_compat(device):
    """Test backward compat: enable_frame_num=True behaves like 'scalar'."""
    skip_if_m60()
    sequence_length = 5
    stride = 1
    batch_size = 2
    filenames = cfr_files

    @pipeline_def
    def bool_pipe():
        _, frame_idx = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num=True,
        )
        return frame_idx

    @pipeline_def
    def scalar_pipe():
        _, frame_idx = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num="scalar",
        )
        return frame_idx

    pipe_bool = bool_pipe(batch_size=batch_size, num_threads=2, device_id=0, seed=42)
    pipe_str = scalar_pipe(batch_size=batch_size, num_threads=2, device_id=0, seed=42)
    compare_pipelines(pipe_bool, pipe_str, batch_size=batch_size, N_iterations=2)


@cartesian_params(["cpu", "gpu"], [False, None])
def test_enable_frame_num_false_none_compat(device, value):
    """Test backward compat: enable_frame_num=False/None behaves like 'none' (no output)."""
    skip_if_m60()
    sequence_length = 5
    stride = 1
    batch_size = 2
    filenames = cfr_files

    @pipeline_def
    def no_frame_num_pipe():
        videos = fn.experimental.readers.video(
            filenames=filenames,
            device=device,
            sequence_length=sequence_length,
            stride=stride,
            enable_frame_num=value,
        )
        return videos

    pipe = no_frame_num_pipe(batch_size=batch_size, num_threads=2, device_id=0, seed=42)
    (videos,) = pipe.run()
    assert videos.as_cpu().at(0).shape[0] == sequence_length, (
        f"enable_frame_num={value!r}: expected {sequence_length} frames, "
        f"got {videos.as_cpu().at(0).shape[0]}"
    )


def test_video_index_reuse():
    batch_size = 2
    sequence_length = 8
    initial_prefetch_size = 16
    data_root_dir = get_dali_extra_path()
    video_files = glob.glob(f"{data_root_dir}/db/video/sintel/video_files/*.mp4")

    @pipeline_def
    def video_pipe(filenames):
        videos = fn.experimental.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            initial_fill=initial_prefetch_size,
        )
        return videos

    pipe = video_pipe(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        filenames=video_files,
        seed=123456,
    )

    # recreate the pipeline to reuse indices build already
    pipe = video_pipe(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        filenames=video_files,
        seed=123456,
    )
    pipe.run()


@cartesian_params(
    # sintel_trailer-720p_3.mp4 has an MP4 edit list that hides its first frames, _0 does not
    ("sintel_trailer-720p_0.mp4", "sintel_trailer-720p_3.mp4"),
    (0, 40, 100),
)
def test_video_decoder_matches_reader(filename, start_frame, sequence_length=8):
    """The decoder (with the default build_index=True) returns the same frames as the reader"""
    path = os.path.join(get_dali_extra_path(), "db", "video", "sintel", "video_files", filename)
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as file_list:
        file_list.write(f"{path} 0 {start_frame} {start_frame + sequence_length}\n")
        file_list.flush()

        @pipeline_def(batch_size=1, num_threads=1, device_id=0)
        def reader_pipe():
            frames, _ = fn.experimental.readers.video(
                device="gpu",
                file_list=file_list.name,
                file_list_format="frames",
                file_list_include_end=False,
                sequence_length=sequence_length,
            )
            return frames

        @pipeline_def(batch_size=1, num_threads=1, device_id=0)
        def decoder_pipe():
            encoded, _ = fn.readers.file(files=[path])
            return fn.decoders.video(
                encoded, device="mixed", start_frame=start_frame, sequence_length=sequence_length
            )

        reader = reader_pipe()
        decoder = decoder_pipe()
        reader.build()
        decoder.build()
        expected = np.array(reader.run()[0].as_cpu().at(0))
        decoded = np.array(decoder.run()[0].as_cpu().at(0))
    assert decoded.shape == expected.shape, f"{decoded.shape} != {expected.shape}"
    diff = np.abs(decoded.astype(np.int16) - expected.astype(np.int16))
    assert np.array_equal(decoded, expected), (
        f"{filename}, start_frame={start_frame}: frame {np.argmax(diff.max(axis=(1, 2, 3)))} "
        f"differs from the reader's, max abs diff {diff.max()}"
    )


class Codec(Enum):
    H264 = "h264"
    HEVC = "hevc"


class NonIdrOp(Enum):
    READER = "reader"
    DECODER = "decoder"


# Streams with open GOPs, H.264 and HEVC: only the first picture is an IDR picture, and the
# keyframes at frames 12, 24, 36, 48 and 60 are non-IDR I-pictures (H.264, with a recovery point
# SEI) or CRA pictures (HEVC) that the container flags as keyframes. Some pictures that follow them
# in the decoding order precede them in the display order and refer to the previous GOP: the RASL
# pictures of every CRA picture in HEVC, and in H.264 the B-pictures before the keyframes at 12
# (10, 11), 36 (35) and 60 (57-59); the H.264 keyframes at 24 and 48 have no such pictures.
# DALI's frame index treats CRA pictures as keyframes, but only IDR pictures in H.264, so the
# H.264 stream is only ever decoded from frame 0: the H.264 tests make sure that it stays so.
non_idr_videos = {
    codec: os.path.join(
        get_dali_extra_path(), "db", "video", "non_idr_keyframes", f"open_gop_{codec.value}.mp4"
    )
    for codec in Codec
}
non_idr_num_frames = 64
non_idr_keyframes = [12, 24, 36, 48, 60]
non_idr_codecs = list(Codec)
AVCOL_SPC_BT709 = 1  # libavutil/pixfmt.h
AVCOL_RANGE_JPEG = 2  # libavutil/pixfmt.h


def nal_unit_types(packet, codec):
    """NAL unit types of a packet in the MP4 (AVCC / HVCC) format: 4-byte length-prefixed units"""
    data = bytes(packet)
    types, pos = [], 0
    while pos + 4 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        if size == 0 or pos + size > len(data):
            break
        types.append(data[pos] & 0x1F if codec == Codec.H264 else (data[pos] >> 1) & 0x3F)
        pos += size
    return types


def is_idr(codec, types):
    """IDR (5) in H.264, IDR_W_RADL and IDR_N_LP (19, 20) in HEVC"""
    return any(t == 5 for t in types) if codec == Codec.H264 else any(t in (19, 20) for t in types)


def non_idr_reader_sequences(codec, starts, sequence_length):
    """The sequences that start at the given frames, read in this order from the file by one reader
    (batch of 1, so the reader's decoder has decoded the previous sequences already)"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as file_list:
        for start in starts:
            # the label is the start frame, to check the order
            file_list.write(f"{non_idr_videos[codec]} {start} {start} {start + sequence_length}\n")
        file_list.flush()

        @pipeline_def(batch_size=1, num_threads=1, device_id=0)
        def reader_pipe():
            frames, labels = fn.experimental.readers.video(
                device="gpu",
                file_list=file_list.name,
                file_list_format="frames",
                file_list_include_end=False,
                sequence_length=sequence_length,
            )
            return frames, labels

        pipe = reader_pipe()
        pipe.build()
        sequences = []
        for start in starts:
            frames, labels = pipe.run()
            label = int(np.array(labels.as_cpu().at(0)).reshape(-1)[0])
            assert label == start, f"The sequence of frame {label} came instead of {start}"
            sequences.append(np.array(frames.as_cpu().at(0)))
        return sequences


def non_idr_reader_frames(codec, start_frame, sequence_length):
    return non_idr_reader_sequences(codec, [start_frame], sequence_length)[0]


def non_idr_decoder_frames(codec, start_frame, sequence_length):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def decoder_pipe():
        encoded, _ = fn.readers.file(files=[non_idr_videos[codec]])
        # the index is what the decoder seeks with
        return fn.decoders.video(
            encoded,
            device="mixed",
            start_frame=start_frame,
            sequence_length=sequence_length,
            build_index=True,
        )

    pipe = decoder_pipe()
    pipe.build()
    return np.array(pipe.run()[0].as_cpu().at(0))


def yuv420_to_rgb_bt601(y, u, v):
    """YUV 4:2:0 to RGB the way the GPU video decoder does it: BT.601, limited range, bilinear
    chroma upsampling.

    The offsets are the exact 8-bit ones (16 for Y, 128 for Cb and Cr): the decoder normalizes the
    8-bit samples by 256 so that 1/16 and 1/2 are 16 and 128 levels."""
    size = (y.shape[1], y.shape[0])
    u = cv2.resize(u, size, interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, size, interpolation=cv2.INTER_LINEAR)
    y = (y - 16) * (255 / 219)
    u = (u - 128) * (255 / 224)
    v = (v - 128) * (255 / 224)
    kr, kb = 0.299, 0.114
    r = y + 2 * (1 - kr) * v
    b = y + 2 * (1 - kb) * u
    g = (y - kr * r - kb * b) / (1 - kr - kb)
    return np.clip(np.rint(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)


@lru_cache(maxsize=None)
def non_idr_reference_frames(codec):
    """All frames of the stream: the YUV frames decoded by FFmpeg, converted to RGB like the GPU
    video decoder does it.

    The conversion is done here and not by FFmpeg (as in extract_frames_from_video) because the
    clips are synthetic (sharp colour edges) and FFmpeg's chroma upsampling differs from the GPU
    decoder's there: 6-16% of the values differ by more than 2 levels, which makes a tolerance
    hide anything below that. With the same conversion only the floating point rounding differs.
    """
    container = av.open(non_idr_videos[codec])
    frames = []
    for frame in container.decode(container.streams.video[0]):
        h, w = frame.height, frame.width
        planes = frame.to_ndarray(format="yuv420p")
        y = planes[:h].astype(np.float32)
        u = planes[h : h + h // 4].reshape(h // 2, w // 2).astype(np.float32)
        v = planes[h + h // 4 :].reshape(h // 2, w // 2).astype(np.float32)
        frames.append(yuv420_to_rgb_bt601(y, u, v))
    return np.stack(frames)


def assert_non_idr_frames_equal(frames, expected, first_frame, what):
    """The frames are bit-identical to the ones decoded in order"""
    assert frames.shape == expected.shape, f"{what}: {frames.shape} != {expected.shape}"
    for i in range(len(expected)):
        if not np.array_equal(frames[i], expected[i]):
            diff = np.abs(frames[i].astype(np.int16) - expected[i].astype(np.int16))
            raise AssertionError(
                f"{what}: frame {first_frame + i} differs from the one decoded in order, "
                f"max abs diff {diff.max()}"
            )


@lru_cache(maxsize=None)
def non_idr_sequential_frames(codec):
    """All frames of the stream, decoded in order from the start"""
    return non_idr_reader_frames(codec, 0, non_idr_num_frames)


@params(*non_idr_codecs)
def test_non_idr_keyframes_fixture(codec):
    """The test video has the keyframes that these tests are about"""
    with av.open(non_idr_videos[codec]) as container:
        stream = container.streams.video[0]
        packets = [p for p in container.demux(stream) if p.size > 0]
        colorspace, color_range = stream.codec_context.colorspace, stream.codec_context.color_range
    assert len(packets) == non_idr_num_frames, f"{len(packets)} != {non_idr_num_frames} packets"
    # the display index of each packet, in the decoding order
    display = {pts: i for i, pts in enumerate(sorted(p.pts for p in packets))}
    frame_ids = [display[p.pts] for p in packets]
    types = [nal_unit_types(p, codec) for p in packets]
    keyframes = [i for i, p in enumerate(packets) if p.is_keyframe]
    idr = [i for i in keyframes if is_idr(codec, types[i])]
    idr_frames = [frame_ids[i] for i in idr]
    assert idr_frames == [0], f"Expected one IDR picture, at frame 0, got {idr_frames}"
    non_idr = [i for i in keyframes if i not in idr]
    assert [frame_ids[i] for i in non_idr] == non_idr_keyframes, (
        f"Expected keyframes that are not IDR pictures at frames {non_idr_keyframes}, "
        f"got {[frame_ids[i] for i in non_idr]}"
    )
    # the pictures that follow a keyframe in the decoding order and precede it in the display
    # order (they refer to the previous GOP)
    leading = {frame_ids[i]: [f for f in frame_ids[i + 1 :] if f < frame_ids[i]] for i in non_idr}
    if codec == Codec.HEVC:
        assert all(21 in types[i] for i in non_idr), "Expected CRA pictures"
        assert all(leading.values()), f"Expected RASL pictures after each CRA, got {leading}"
        assert any(set(t) & {8, 9} for t in types), "Expected RASL_N or RASL_R pictures"
    else:
        assert any(leading.values()), f"Expected leading pictures, got {leading}"
    # non_idr_reference_frames converts with the BT.601 matrix and the limited range, which is
    # what the GPU decoder does unless the stream is full range (or, possibly, tagged as BT.709)
    assert color_range != AVCOL_RANGE_JPEG, f"{codec}: the test clip is full range"
    assert colorspace != AVCOL_SPC_BT709, f"{codec}: the test clip is tagged as BT.709"


@params(*non_idr_codecs)
def test_non_idr_keyframes_sequential_decoding(codec):
    """Decoding the stream in order gives all the frames, as decoded by FFmpeg"""
    skip_if_m60()
    frames = non_idr_sequential_frames(codec)
    reference = non_idr_reference_frames(codec)
    assert len(frames) == len(reference), f"{len(frames)} != {len(reference)} frames"
    # the frames match the reference up to the floating point rounding of the colour conversion,
    # so a corrupted block, however small, is caught, and so is a frame swapped with another (any
    # two frames of the clips differ by far more than that somewhere)
    diff = np.abs(frames.astype(np.int16) - reference.astype(np.int16)).reshape(len(frames), -1)
    for i, frame_diff in enumerate(diff.max(axis=1)):
        assert frame_diff <= 1, f"{codec}: frame {i} differs from FFmpeg's by {frame_diff} levels"


@cartesian_params(
    non_idr_codecs,
    list(NonIdrOp),
    # around the keyframes that are not IDR pictures (in HEVC the frame before each of them is a
    # RASL picture), and in the first GOP; the sequences have 4 frames
    sorted(
        {0, 5}
        | {k + d for k in non_idr_keyframes for d in (-1, 0, 1) if k + d + 4 <= non_idr_num_frames}
    ),
)
def test_non_idr_keyframes_random_access(codec, op, start_frame, sequence_length=4):
    """Decoding from any frame gives the same frames as decoding the stream in order"""
    skip_if_m60()
    decode = non_idr_reader_frames if op == NonIdrOp.READER else non_idr_decoder_frames
    frames = decode(codec, start_frame, sequence_length)
    expected = non_idr_sequential_frames(codec)[start_frame : start_frame + sequence_length]
    assert_non_idr_frames_equal(frames, expected, start_frame, f"{codec} {op}")


# The reader keeps its decoder open while the next sequence is from the same file, so seeking can
# start from a decoder that has decoded other sequences already (the decoder operator creates a new
# one for each sample). In HEVC, the order makes it seek: far forward to a leading picture of a
# keyframe (58), backwards (33, 35, 20, 0, 23), backwards by one frame to the keyframe (36), forward
# by a few frames, which only skips the frames in between (26, 11, 24), and forward by many (45,
# 60). The H.264 stream has no keyframe to seek to after frame 0 (see above), so there the reader
# skips forward and goes back to frame 0.
non_idr_warm_starts = [58, 33, 36, 35, 20, 26, 45, 0, 11, 24, 23, 60]


@params(*non_idr_codecs)
def test_non_idr_keyframes_warm_seeks(codec, sequence_length=4):
    """Seeking on a decoder that has decoded other sequences gives the frames of the stream"""
    skip_if_m60()
    sequences = non_idr_reader_sequences(codec, non_idr_warm_starts, sequence_length)
    expected = non_idr_sequential_frames(codec)
    for start, frames in zip(non_idr_warm_starts, sequences, strict=True):
        assert_non_idr_frames_equal(
            frames, expected[start : start + sequence_length], start, f"{codec}, sequence {start}"
        )


@functools.lru_cache(maxsize=4)
def _reference_frames(filename):
    """All frames of the file, decoded with PyAV, in presentation order (the last few files are
    kept: the tests check one file after another)."""
    with av.open(filename) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def _assert_frames_match_file(frames, frame_nums, filename, window=32):
    """Checks that frames[i] is the frame frame_nums[i] of the file (negative: padding, skipped).

    Decoders differ slightly in color conversion, so a frame must be close to the reference and
    closer to it than to any other frame up to `window` frames away (None: of the whole file)."""
    reference = _reference_frames(filename)
    for frame, num in zip(frames, frame_nums, strict=True):
        if num < 0:
            continue
        start = 0 if window is None else max(0, num - window)
        end = len(reference) if window is None else min(len(reference), num + window + 1)
        diffs = {i: np.abs(frame.astype(np.int32) - reference[i]).mean() for i in range(start, end)}
        closest = min(diffs, key=diffs.get)
        assert (
            diffs[num] < 10
        ), f"{filename}: frame {num} differs from the reference by {diffs[num]}"
        assert (
            diffs[num] <= diffs[closest] + 0.5
        ), f"{filename}: frame {num} is closer to frame {closest}"


# The reader decodes the samples of a batch in parallel, by up to num_threads decoders (decode
# lanes). Its output must not depend on how many there are.
_lanes_reader_cases = [
    dict(sequence_length=5),  # consecutive sequences of each file
    dict(sequence_length=4, stride=3, step=20),  # sparse samples of each file
    dict(sequence_length=5, step=2),  # overlapping samples
    dict(sequence_length=6, random_shuffle=True, initial_fill=8),
    dict(sequence_length=6, uniform_sample=True),
    dict(sequence_length=7, pad_mode="constant", fill_value=[30, 60, 90], enable_timestamps=True),
    dict(sequence_length=5, pad_last_batch=True, num_shards=3, shard_id=2),
]


_sintel_dir = os.path.join(get_dali_extra_path(), "db", "video", "sintel")

# H.264 clips that start with packets hidden by an edit list (62 and 16), which are decoded before
# the first frame, and after a seek that lands on them
lead_in_files = [
    os.path.join(_sintel_dir, "labelled_videos", "0", "sintel_trailer-720p_6.mp4"),
    os.path.join(_sintel_dir, "labelled_videos", "2", "sintel_trailer-720p_5.mp4"),
]


def _lanes_reader_files(device):
    # H.264 is not enabled on the CPU backend
    gpu_files = codec_files["h264"] + lead_in_files
    return cfr_files + vfr_files + (gpu_files if device == "gpu" else [])


def _check_reader_output_independent_of_num_threads(
    device, kwargs, batch_size, num_threads, files=None, window=32
):
    files = files or _lanes_reader_files(device)

    @pipeline_def(batch_size=batch_size, device_id=0, seed=42)
    def pipe():
        # videos, labels (the index of the file), frame numbers and, with enable_timestamps,
        # timestamps
        return tuple(
            fn.experimental.readers.video(
                filenames=files,
                labels=list(range(len(files))),
                device=device,
                enable_frame_num="sequence",
                **kwargs,
            )
        )

    compare_pipelines(
        pipe(num_threads=1), pipe(num_threads=num_threads), batch_size=batch_size, N_iterations=8
    )

    # The two runs could agree on wrong frames, so check the frames against PyAV as well
    p = pipe(num_threads=num_threads)
    for _ in range(4):
        videos, labels, frame_nums, *_ = (t.as_cpu() for t in p.run())
        for i in range(batch_size):
            _assert_frames_match_file(
                np.array(videos[i]),
                np.array(frame_nums[i]),
                files[int(np.array(labels[i])[0])],
                window,
            )


@cartesian_params(["cpu", "gpu"], range(len(_lanes_reader_cases)))
def test_reader_output_independent_of_num_threads(device, case):
    skip_if_m60()
    _check_reader_output_independent_of_num_threads(
        device, _lanes_reader_cases[case], batch_size=4, num_threads=4
    )


# One sample per batch: one lane decodes, whatever the number of threads, and its decoder carries
# over from batch to batch
@cartesian_params(["cpu", "gpu"], (0, 1, 3))
def test_reader_batch_size_1_output_independent_of_num_threads(device, case):
    skip_if_m60()
    _check_reader_output_independent_of_num_threads(
        device, _lanes_reader_cases[case], batch_size=1, num_threads=4
    )


# An H.264 clip with a long lead-in hidden by an edit list (188 pictures). The GPU decoder decodes
# those pictures, but does not map or convert them, nor the pictures that it only skips
long_lead_in_files = [
    os.path.join(_sintel_dir, "video_files", "sintel_trailer-720p_4.mp4"),
]


# Consecutive sequences (from the start of a file, after its lead-in), sparse samples (frames
# skipped in between) and shuffled samples (seeks, also back to the lead-in)
@params(0, 1, 3)
def test_reader_lead_in_output_independent_of_num_threads(case):
    skip_if_m60()
    _check_reader_output_independent_of_num_threads(
        "gpu",
        _lanes_reader_cases[case],
        batch_size=4,
        num_threads=4,
        files=long_lead_in_files,
        window=16,
    )


# With fewer lanes than samples, a lane decodes several runs, of one file or more, in an order
# picked from their estimated work: tails padded with the last frame (decoding it again reopens
# the file) and uniformly sampled frames
_multi_run_lanes_cases = [
    dict(sequence_length=7, pad_mode="edge", random_shuffle=True, initial_fill=16),
    dict(sequence_length=7, step=4, pad_mode="edge"),
    dict(sequence_length=6, uniform_sample=True),
]


@cartesian_params(["cpu", "gpu"], range(len(_multi_run_lanes_cases)))
def test_reader_multi_run_lanes_output_independent_of_num_threads(device, case):
    skip_if_m60()
    _check_reader_output_independent_of_num_threads(
        device, _multi_run_lanes_cases[case], batch_size=8, num_threads=2
    )


# 26 samples in 3 shards (pad_last_batch): shard 0 has samples 0-7, padded to 9 samples, so that
# its third batch is sample 7, four times (decoded in the batch before it); shard 2 has samples
# 17-25, so that its third batch is sample 25 and three repeats of it
@cartesian_params(("cpu", "gpu"), (0, 2))
def test_reader_pad_last_batch_repeats_last_sample(device, shard_id):
    """With pad_last_batch, a shard's last sample is repeated, in its own batch or as the whole
    next batch: the same sample, which must not be decoded again while an earlier batch may still
    be in use."""
    skip_if_m60()
    batch_size = 4
    files = _lanes_reader_files(device)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file_list:
        for i in range(26):
            start = (i // len(files)) * 4
            file_list.write(f"{files[i % len(files)]} {i} {start} {start + 4}\n")

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, prefetch_queue_depth=3)
    def pipe():
        videos, labels = fn.experimental.readers.video(
            file_list=file_list.name,
            file_list_format="frames",
            file_list_include_end=False,
            device=device,
            sequence_length=4,
            pad_last_batch=True,
            num_shards=3,
            shard_id=shard_id,
        )
        return videos, labels

    try:
        p = pipe()
        samples = []
        for _ in range(3):
            videos, labels = p.run()
            videos = videos.as_cpu() if device == "gpu" else videos
            labels = labels.as_cpu() if device == "gpu" else labels
            samples += [
                (np.array(videos[i]), int(np.array(labels[i])[0])) for i in range(batch_size)
            ]
    finally:
        os.remove(file_list.name)
    first, end = 26 * shard_id // 3, 26 * (shard_id + 1) // 3
    num_read = end - first
    assert [label for _, label in samples] == list(range(first, end)) + [end - 1] * (12 - num_read)
    for video, _ in samples[num_read:]:
        np.testing.assert_array_equal(video, samples[num_read - 1][0])
    for video, label in samples[:num_read]:
        start = (label // len(files)) * 4
        _assert_frames_match_file(video, range(start, start + 4), files[label % len(files)])


def _num_frames(filename, device):
    """The frames of a file, as the reader counts them."""

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe():
        return fn.experimental.readers.video(
            filenames=[filename], sequence_length=1, device=device, name="Reader"
        )

    p = pipe()
    p.build()
    return p.reader_meta("Reader")["epoch_size"]


@cartesian_params(("cpu", "gpu"), (1, 4))
def test_reader_reports_first_error_of_batch(device, num_threads):
    """Samples fail in two lanes of a batch and in the next batch: the error is the first one of
    the batch, at that batch, whatever the lanes' timing."""
    skip_if_m60()
    batch_size = 4
    files = cfr_files  # with different numbers of frames
    num_frames = [_num_frames(f, device) for f in files]
    assert num_frames[0] != num_frames[1]
    # (file, start) of each sample, with sequences that start past the end of the file
    samples = [(0, 0), (1, 0), (0, 4), (1, 4)]  # batch 0
    samples += [(0, 8), (0, num_frames[0] + 10), (1, 8), (1, num_frames[1] + 20)]  # batch 1
    samples += [(0, num_frames[0] + 30), (1, 12), (0, 12), (1, 16)]  # batch 2
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as file_list:
        for i, (f, start) in enumerate(samples):
            file_list.write(f"{files[f]} {i} {start} {start + 4}\n")

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def pipe():
        videos, _ = fn.experimental.readers.video(
            file_list=file_list.name,
            file_list_format="frames",
            file_list_include_end=False,
            device=device,
            sequence_length=4,
        )
        return videos

    try:
        p = pipe()
        p.run()  # batch 0
        expected = rf"frame index {num_frames[0] + 10} .*\[0, {num_frames[0]}\)"
        with assert_raises(RuntimeError, regex=expected):
            p.run()
    finally:
        os.remove(file_list.name)
